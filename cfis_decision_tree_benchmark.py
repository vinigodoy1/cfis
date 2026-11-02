# CFIS final Decision Tree benchmark — single-cell compatible Python script
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import os, json, time, hashlib, warnings, threading
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.stats import spearmanr, kendalltau
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, balanced_accuracy_score, r2_score, mean_squared_error, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

ROOT=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
DATA=ROOT/'01_Datasets'
RUN=ROOT/'04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1'
MANIFEST=RUN/'Manifests/inner_benchmark_manifest.csv'
CKPT=RUN/'Checkpoints'; OUT=RUN/'Results'
OUT.mkdir(parents=True,exist_ok=True)
SEED=42
# Set to a dataset name for controlled execution, or None for all twelve.
ONLY_DATASET=None
# Set to an integer 1..10 for a controlled split, or None for all ten.
ONLY_SPLIT=None
# Set True to continue other datasets if a particular dataset fails.
CONTINUE_ON_ERROR=True
# Dense conversion safety cap (GiB). Set higher only if your Colab RAM permits.
MAX_DENSE_GIB=3.0

with open(RUN/'Manifests/protocol.json',encoding='utf-8') as f: protocol=json.load(f)
assert protocol['model']=='DecisionTree' and protocol['seed']==SEED
assert 'ANOVA_or_F_Regression' not in protocol['methods']
manifest=pd.read_csv(MANIFEST)
assert len(manifest)==8000
assert manifest['Protocol_SHA256'].nunique()==1
assert manifest['Protocol_SHA256'].iloc[0]==protocol['protocol_sha256']
METHODS=protocol['methods']; PH=protocol['protocol_sha256']
assert len(METHODS)==10

# The original CFIS function and its validated numerical adapter must be loaded
# in the notebook before executing this script.
needed=['cfis_function','cfis_globals','original_permutation','consolidated_permutation_adapter']
missing=[n for n in needed if n not in globals()]
if missing: raise RuntimeError('CFIS ausente no notebook: '+', '.join(missing))
if globals().get('exact_cfis') is not True or globals().get('equivalent_1e12') is not True:
    raise RuntimeError('Execute primeiro a validação de equivalência do CFIS (exact_cfis/equivalent_1e12).')
if cfis_function.__globals__ is not cfis_globals: raise RuntimeError('Namespace CFIS incompatível.')
if cfis_globals.get('permutation_importance') is not original_permutation:
    raise RuntimeError('Estado inicial da permutação do CFIS incompatível.')

try:
    import psutil
    PROCESS=psutil.Process(os.getpid())
except ImportError:
    PROCESS=None

# Atomic checkpoints; no checkpoint is accepted from another protocol.
def save_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def read_json(path):
    with open(path,encoding='utf-8') as f: obj=json.load(f)
    if obj.get('protocol_sha256')!=PH: raise RuntimeError('Checkpoint de outro protocolo: '+str(path))
    return obj

class Measure:
    # RSS sampled during execution; this is a sampled peak, not hardware peak.
    def __enter__(self):
        self.t0=time.perf_counter(); self.c0=time.process_time()
        self.stop=threading.Event(); self.peak=0
        if PROCESS:
            def poll():
                while not self.stop.is_set():
                    try: self.peak=max(self.peak,PROCESS.memory_info().rss)
                    except Exception: pass
                    self.stop.wait(.05)
            self.thread=threading.Thread(target=poll,daemon=True)
            self.thread.start()
        return self
    def __exit__(self,*args):
        self.wall=time.perf_counter()-self.t0
        self.cpu=time.process_time()-self.c0
        self.stop.set()
        if PROCESS:
            self.thread.join(timeout=1)
            try: self.peak=max(self.peak,PROCESS.memory_info().rss)
            except Exception: pass
        self.rss_mib=self.peak/(1024**2) if self.peak else None

def dense_checked(X):
    n,p=X.shape
    size=n*p*8/1024**3
    if size>MAX_DENSE_GIB: raise MemoryError(f'Dense float64 estimado em {size:.2f} GiB, limite {MAX_DENSE_GIB:.2f} GiB')
    if sparse.issparse(X): X=X.toarray()
    return np.asarray(X,dtype=np.float64)

def order(scores):
    a=np.asarray(scores,dtype=float).reshape(-1)
    a=np.nan_to_num(a,nan=-np.inf,neginf=-np.inf,posinf=np.inf)
    return np.argsort(-a,kind='stable').tolist()

def mi_scores(X,y,task):
    fn=mutual_info_classif if task=='classification' else mutual_info_regression
    return fn(X,y,random_state=SEED)

def tree(task):
    return DecisionTreeClassifier(random_state=SEED) if task=='classification' else DecisionTreeRegressor(random_state=SEED)

def l1_ranking(X,y,task):
    if task=='regression':
        est=Lasso(alpha=.01,max_iter=5000,random_state=SEED)
        est.fit(X,y); scores=np.abs(est.coef_)
    elif len(np.unique(y))>2:
        est=OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear',C=1,max_iter=5000,random_state=SEED))
        est.fit(X,y); scores=np.mean(np.abs(np.vstack([e.coef_.reshape(-1) for e in est.estimators_])),axis=0)
    else:
        est=LogisticRegression(penalty='l1',solver='liblinear',C=1,max_iter=5000,random_state=SEED)
        est.fit(X,y); scores=np.abs(est.coef_).mean(axis=0)
    return order(scores)

def discretize(X):
    # Fixed quantile discretization, explicitly used for the mRMR MID variant.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kb=KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='quantile')
        return np.asarray(kb.fit_transform(X),dtype=np.int16)

def mrmr_ranking(X,y,task):
    # Greedy MID: MI(feature,target) - mean MI(feature,selected).
    # Pairwise redundancy is estimated from discretized continuous features.
    rel=np.asarray(mi_scores(X,y,task),dtype=float)
    Z=discretize(X); p=X.shape[1]
    # Build symmetric MI redundancy once. For high-dimensional datasets this is expensive.
    redundancy=np.zeros((p,p),dtype=np.float32)
    for i in range(p):
        for j in range(i+1,p):
            val=mutual_info_score(Z[:,i],Z[:,j])
            redundancy[i,j]=redundancy[j,i]=val
    selected=[]; remaining=np.ones(p,dtype=bool)
    redsum=np.zeros(p,dtype=float)
    for step in range(p):
        score=rel-redsum/max(step,1) if step else rel
        score[~remaining]=-np.inf
        idx=int(np.argmax(score)); selected.append(idx); remaining[idx]=False
        redsum+=redundancy[:,idx]
    return selected

def cfs_ranking(X,y,task):
    # Hall-style correlation merit + greedy forward ordering.
    # Multiclass relevance: max absolute Pearson corr against one-vs-rest targets.
    p=X.shape[1]
    with np.errstate(divide='ignore',invalid='ignore'):
        corr=np.abs(np.corrcoef(X,rowvar=False))
    corr=np.nan_to_num(corr,nan=0.,posinf=0.,neginf=0.)
    np.fill_diagonal(corr,0)
    if task=='classification':
        classes=np.unique(y)
        targets=np.column_stack([(y==c).astype(float) for c in classes])
    else: targets=np.asarray(y,dtype=float).reshape(-1,1)
    Xc=X-X.mean(axis=0); Tc=targets-targets.mean(axis=0)
    denom=np.sqrt((Xc**2).sum(axis=0)[:,None]*(Tc**2).sum(axis=0)[None,:])
    relevance=np.divide(np.abs(Xc.T@Tc),denom,out=np.zeros((p,targets.shape[1])),where=denom>0).max(axis=1)
    selected=[]; remaining=np.ones(p,dtype=bool)
    relsum=0.; redpairs=0.
    for step in range(p):
        candidates=np.flatnonzero(remaining)
        k=step+1
        rel_total=relsum+relevance[candidates]
        if step:
            red_total=redpairs+2*corr[np.ix_(candidates,selected)].sum(axis=1)
        else: red_total=np.zeros(len(candidates))
        merit=rel_total/np.sqrt(k+red_total)
        idx=int(candidates[np.argmax(merit)])
        if step: redpairs+=2*corr[idx,selected].sum()
        relsum+=relevance[idx]; selected.append(idx); remaining[idx]=False
    return selected

def shap_ranking(X,y,task):
    import shap
    est=tree(task); est.fit(X,y)
    vals=np.asarray(shap.TreeExplainer(est).shap_values(X))
    p=X.shape[1]
    if vals.ndim==2 and vals.shape[1]==p: scores=np.abs(vals).mean(axis=0)
    elif vals.ndim==3 and vals.shape[1]==p: scores=np.abs(vals).mean(axis=(0,2))
    elif vals.ndim==3 and vals.shape[2]==p: scores=np.abs(vals).mean(axis=(0,1))
    else: raise RuntimeError('Formato SHAP inesperado: '+str(vals.shape))
    return order(scores)

def pfi_ranking(X,y,task):
    a,b=train_test_split(np.arange(len(y)),test_size=.25,random_state=SEED,
        stratify=y if task=='classification' else None)
    est=tree(task); est.fit(X[a],y[a])
    score='balanced_accuracy' if task=='classification' else 'r2'
    imp=permutation_importance(est,X[b],y[b],scoring=score,n_repeats=10,random_state=SEED,n_jobs=1)
    return order(imp.importances_mean)

def cfis_ranking(X,y,names,task):
    try:
        cfis_globals['permutation_importance']=consolidated_permutation_adapter
        result=cfis_function(pd.DataFrame(X,columns=names),pd.Series(y),task=task,
                             model=None,n_repeats=10,random_state=SEED)
    finally:
        cfis_globals['permutation_importance']=original_permutation
    if not isinstance(result,pd.DataFrame) or 'Feature' not in result:
        raise RuntimeError('Saída CFIS sem coluna Feature')
    mapping={n:i for i,n in enumerate(names)}
    return [mapping[n] for n in result['Feature'].astype(str)]

def compute_ranking(method,X,y,names,task,split_no):
    p=X.shape[1]
    if method=='CFIS': return cfis_ranking(X,y,names,task)
    if method=='Mutual_Information': return order(mi_scores(X,y,task))
    if method=='mRMR': return mrmr_ranking(X,y,task)
    if method=='L1': return l1_ranking(X,y,task)
    if method=='CFS': return cfs_ranking(X,y,task)
    if method=='RFE':
        r=RFE(estimator=tree(task),n_features_to_select=1,step=1).fit(X,y)
        return np.argsort(r.ranking_,kind='stable').tolist()
    if method=='PFI': return pfi_ranking(X,y,task)
    if method=='TreeSHAP': return shap_ranking(X,y,task)
    if method=='MDI':
        est=tree(task).fit(X,y)
        return order(est.feature_importances_)
    if method=='Random_TopK':
        return np.random.default_rng(SEED+split_no).permutation(p).tolist()
    raise ValueError(method)

def load_data(ds):
    source=next((DATA/folder/f'{ds}.parquet' for folder in ('Raw','Synthetic') if (DATA/folder/f'{ds}.parquet').exists()),None)
    if source is None: raise FileNotFoundError(ds)
    with open(source.with_name(f'{ds}_metadata.json'),encoding='utf-8') as f: meta=json.load(f)
    frame=pd.read_parquet(source)
    idx=np.load(DATA/'Splits'/f'{ds}_train_indices.npy',allow_pickle=False)
    outer=frame.iloc[idx]
    target=meta['target_column']
    y=outer[target].to_numpy().ravel()
    X=outer.drop(columns=[target])
    if ds=='20_Newsgroups':
        if 'text' not in X.columns: raise RuntimeError('Coluna text ausente em 20_Newsgroups')
        X=X['text'].fillna('').astype(str)
    return X,y

def get_names(prep,variance,Xfit,p):
    try: names=np.asarray(prep.get_feature_names_out(),dtype=str)
    except (AttributeError,ValueError):
        if isinstance(Xfit,pd.DataFrame): names=np.asarray(Xfit.columns,dtype=str)
        else: names=np.asarray([f'f_{i}' for i in range(p)],dtype=str)
    mask=variance.get_support()
    if len(names)!=len(mask): raise RuntimeError(f'Nomes antes do filtro: {len(names)}, máscara: {len(mask)}')
    names=names[mask]
    if len(set(names))!=len(names): raise RuntimeError('Nomes de características duplicados')
    return names

def eval_job(ds,split_no,method,k,rank,features,Xfit,yfit,Xval,yval,task,folder):
    path=folder/f'prediction__{method}__K{k}.json'
    if path.exists(): read_json(path); return False
    cols=[features[n] for n in rank[:k]]
    with Measure() as measure:
        model=tree(task)
        model.fit(Xfit[:,cols],yfit)
        pred=model.predict(Xval[:,cols])
    record={'protocol_sha256':PH,'Dataset':ds,'Split':split_no,'Method':method,
            'Model':'DecisionTree','TopK':int(k),'Fit_Wall_Seconds':measure.wall,
            'Fit_CPU_Seconds':measure.cpu,'Fit_Sampled_Peak_RSS_MiB':measure.rss_mib,
            'Estimated_Energy_Wh':None,'Outer_Test_Used':False}
    if task=='classification':
        record.update(Macro_F1=float(f1_score(yval,pred,average='macro',zero_division=0)),
                      Balanced_Accuracy=float(balanced_accuracy_score(yval,pred)))
    else:
        record.update(R2=float(r2_score(yval,pred)),RMSE=float(np.sqrt(mean_squared_error(yval,pred))))
    save_json(path,record)
    return True

errors=[]; total_new_rank=0; total_new_jobs=0
all_ds=protocol['datasets'] if ONLY_DATASET is None else [ONLY_DATASET]
print('='*76)
print('CFIS — BENCHMARK DEFINITIVO DECISION TREE — EXECUÇÃO AUTOMÁTICA')
print('Protocolo:',PH,'| jobs no manifesto:',len(manifest))
print('Datasets:',len(all_ds),'| métodos:',len(METHODS),'| teste externo: BLOQUEADO')
print('Energia: NÃO MEDIDA (campo nulo até instrumentação validada)')
print('='*76,flush=True)

for ds in all_ds:
    try:
        Xraw,y=load_data(ds)
        task='regression' if ds in protocol['regression_datasets'] else 'classification'
        artifact=joblib.load(DATA/'Processed'/ds/'preprocessor.joblib')
        splitter=(StratifiedShuffleSplit if task=='classification' else ShuffleSplit)(
            n_splits=protocol['inner_splits'],test_size=protocol['inner_validation_fraction'],random_state=SEED)
        splits=list(splitter.split(Xraw,y) if task=='classification' else splitter.split(Xraw))
        ks=protocol['topk'][ds]
        print(f'\nDATASET {ds} | task={task} | outer_train={len(y)} | K={ks}',flush=True)
        for sn,(fit_idx,val_idx) in enumerate(splits,1):
            if ONLY_SPLIT is not None and sn!=ONLY_SPLIT: continue
            folder=CKPT/ds/f'split_{sn:02d}'; folder.mkdir(parents=True,exist_ok=True)
            # Skip preprocessing entirely when all checkpoints already exist.
            complete=all((folder/f'ranking__{m}.json').exists() and all(
                (folder/f'prediction__{m}__K{k}.json').exists() for k in ks) for m in METHODS)
            if complete:
                for m in METHODS:
                    read_json(folder/f'ranking__{m}.json')
                    for k in ks: read_json(folder/f'prediction__{m}__K{k}.json')
                print(f'  Split {sn:02d}: 100% reutilizado',flush=True)
                continue
            xfitraw=Xraw.iloc[fit_idx]; xvalraw=Xraw.iloc[val_idx]
            yfit=y[fit_idx]; yval=y[val_idx]
            prep=clone(artifact['preprocessor']); vf=clone(artifact['variance_filter'])
            xfitpre=prep.fit_transform(xfitraw,yfit)
            xvalpre=prep.transform(xvalraw)
            xfit=dense_checked(vf.fit_transform(xfitpre))
            xval=dense_checked(vf.transform(xvalpre))
            names=get_names(prep,vf,xfitraw,xfit.shape[1])
            if len(names)!=xfit.shape[1] or max(ks)>len(names):
                raise RuntimeError(f'K inválido ou nomes incompatíveis: {ds} split {sn}, p={len(names)}')
            mapping={n:i for i,n in enumerate(names)}
            newrank=0; newjobs=0
            for method in METHODS:
                rankpath=folder/f'ranking__{method}.json'
                if rankpath.exists():
                    rr=read_json(rankpath)
                    if rr['feature_names']!=names.tolist():
                        raise RuntimeError(f'Nomes incompatíveis com ranking: {ds}/{sn}/{method}')
                    ranking=rr['ranking_names']
                else:
                    print(f'  Split {sn:02d} | ranking {method}...',flush=True)
                    with Measure() as measure:
                        inds=compute_ranking(method,xfit,yfit,names,task,sn)
                    if len(inds)!=len(names) or sorted(inds)!=list(range(len(names))):
                        raise RuntimeError(f'Ranking inválido: {ds}/{sn}/{method}')
                    ranking=names[inds].tolist()
                    rr={'protocol_sha256':PH,'dataset':ds,'split':sn,'method':method,
                        'feature_names':names.tolist(),'ranking_names':ranking,
                        'selection_wall_seconds':measure.wall,
                        'selection_cpu_seconds':measure.cpu,
                        'selection_sampled_peak_rss_mib':measure.rss_mib,
                        'estimated_energy_wh':None,'outer_test_used':False,
                        'implementation':{
                            'mRMR':'MID, MI discretizada por quantis para redundância',
                            'CFS':'correlação absoluta, mérito CFS, greedy forward',
                            'RFE':'DecisionTree, step=1, ranking completo',
                            'PFI':'DecisionTree, 25% de treino interno reservado para permutação',
                            'TreeSHAP':'TreeExplainer, média de |SHAP|',
                            'L1':'LogisticRegression L1 OvR / Lasso alpha=.01',
                            'Random_TopK':'permutação aleatória determinística, prefixos aninhados'
                        }.get(method,'ranking padrão do método')}
                    save_json(rankpath,rr)
                    newrank+=1; total_new_rank+=1
                if len(ranking)!=len(names) or set(ranking)!=set(names):
                    raise RuntimeError(f'Ranking incompatível: {ds}/{sn}/{method}')
                for k in ks:
                    if eval_job(ds,sn,method,k,ranking,mapping,xfit,yfit,xval,yval,task,folder):
                        newjobs+=1; total_new_jobs+=1
            print(f'  Split {sn:02d} concluído | rankings novos={newrank} | avaliações novas={newjobs}',flush=True)
    except Exception as exc:
        errors.append({'Dataset':ds,'Error':repr(exc)})
        print(f'ERRO em {ds}: {exc!r}',flush=True)
        if not CONTINUE_ON_ERROR: raise

# Consolidate all available checkpoints, including partial results after an error.
rows=[]; ranking_records={}; selection_rows=[]
for ds in protocol['datasets']:
    for sn in range(1,protocol['inner_splits']+1):
        folder=CKPT/ds/f'split_{sn:02d}'
        for method in METHODS:
            rp=folder/f'ranking__{method}.json'
            if not rp.exists(): continue
            rr=read_json(rp)
            ranking_records[(ds,sn,method)]=rr['ranking_names']
            selection_rows.append({'Dataset':ds,'Split':sn,'Method':method,
                'Selection_Wall_Seconds':rr['selection_wall_seconds'],
                'Selection_CPU_Seconds':rr['selection_cpu_seconds'],
                'Selection_Sampled_Peak_RSS_MiB':rr.get('selection_sampled_peak_rss_mib'),
                'Estimated_Energy_Wh':rr.get('estimated_energy_wh')})
            for k in protocol['topk'][ds]:
                path=folder/f'prediction__{method}__K{k}.json'
                if path.exists(): rows.append(read_json(path))

pred=pd.DataFrame(rows); sel=pd.DataFrame(selection_rows)
if len(pred):
    pred.to_csv(OUT/'predictive_internal.csv',index=False)
    primary=np.where(pred['Dataset'].isin(protocol['regression_datasets']),'R2','Macro_F1')
    pred['Primary_Score']=[float(row[m]) for (_,row),m in zip(pred.iterrows(),primary)]
    summary=pred.groupby(['Dataset','Method','TopK'],as_index=False).agg(
        Mean_Score=('Primary_Score','mean'),Std_Score=('Primary_Score','std'),
        Splits=('Split','nunique'),Mean_Fit_Seconds=('Fit_Wall_Seconds','mean'))
    summary.to_csv(OUT/'predictive_summary.csv',index=False)
if len(sel): sel.to_csv(OUT/'selection_efficiency.csv',index=False)

# Inter-method ranking concordance against TreeSHAP as a reference, NOT ground truth.
concordance=[]
for (ds,sn,method),rank in ranking_records.items():
    ref=ranking_records.get((ds,sn,'TreeSHAP'))
    if method=='TreeSHAP' or ref is None or set(rank)!=set(ref): continue
    a={v:i for i,v in enumerate(rank)}; b={v:i for i,v in enumerate(ref)}
    # Positions aligned by feature identity.
    items=sorted(a)
    rho=float(spearmanr([a[v] for v in items],[b[v] for v in items]).statistic)
    tau=float(kendalltau([a[v] for v in items],[b[v] for v in items]).statistic)
    for k in protocol['topk'][ds]:
        A=set(rank[:k]); B=set(ref[:k]); inter=len(A&B)
        concordance.append({'Dataset':ds,'Split':sn,'Method':method,'Reference':'TreeSHAP',
                            'TopK':k,'Jaccard':inter/len(A|B),'Precision_at_K':inter/k,
                            'Spearman':rho,'Kendall':tau})
if concordance: pd.DataFrame(concordance).to_csv(OUT/'concordance_treeshap.csv',index=False)

stability=[]
for ds in protocol['datasets']:
    for method in METHODS:
        rankings=[ranking_records[(ds,s,method)] for s in range(1,11) if (ds,s,method) in ranking_records]
        if len(rankings)<2: continue
        if any(set(r)!=set(rankings[0]) for r in rankings):
            # Feature universe changes after train-only variance filtering; don't silently align.
            print(f'AVISO: estabilidade de {ds}/{method} requer alinhamento por nomes; não calculada.')
            continue
        p=len(rankings[0]); n=len(rankings)
        for k in protocol['topk'][ds]:
            if k>p: continue
            sets=[set(r[:k]) for r in rankings]
            jacc=np.mean([len(a&b)/len(a|b) for a,b in combinations(sets,2)])
            universe=sorted(set(rankings[0])); freq=np.array([sum(v in s for s in sets)/n for v in universe])
            # Sample variance of Bernoulli selection indicators, ddof=1.
            variance=(n/(n-1))*np.mean(freq*(1-freq))
            expected=(k/p)*(1-k/p)
            nog=float(1-variance/expected) if expected>0 else None
            stability.append({'Dataset':ds,'Method':method,'TopK':k,'Splits':n,
                              'Mean_Jaccard':float(jacc),'Nogueira':nog})
if stability: pd.DataFrame(stability).to_csv(OUT/'stability_summary.csv',index=False)

print('\n'+'='*76)
print('RESUMO DA EXECUÇÃO')
print('Rankings novos nesta execução:',total_new_rank)
print('Avaliações novas nesta execução:',total_new_jobs)
print('Avaliações disponíveis:',len(pred),'de 8000')
print('Rankings disponíveis:',len(ranking_records),'de 1200')
print('Datasets com erro:',len(errors))
for err in errors: print(' -',err['Dataset'],err['Error'])
print('Resultados:',OUT)
print('Teste externo utilizado: NÃO')
print('Energia: NÃO MEDIDA')
print('='*76)
