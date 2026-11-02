# CFIS — FASE 5D — CANDIDATO DE EXECUÇÃO EXTERNA (BLOQUEADO)
# Gerado a partir das funções originais do benchmark Decision Tree.
# IMPORTANTE: execução de teste externo PROIBIDA nesta versão.
from pathlib import Path
import json, hashlib, ast, inspect, time, warnings, datetime
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import train_test_split
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
BASE=RUN/'Results/Auditoria_Cientifica'
OUT=BASE/'Fase5D_Candidato_Externo_DT'
SEED=42
MAX_DENSE_GIB=3.0
# NÃO alterar: a execução real só pode ser liberada após auditoria independente.
EXTERNAL_TEST_EXECUTION_ALLOWED=False

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def original_cfis_ready():
    required=['cfis_function','cfis_globals','original_permutation','consolidated_permutation_adapter']
    missing=[n for n in required if n not in globals()]
    if missing: return False,'CFIS não carregado no notebook: '+', '.join(missing)
    if globals().get('exact_cfis') is not True or globals().get('equivalent_1e12') is not True:
        return False,'Validação original de equivalência CFIS não carregada.'
    if cfis_function.__globals__ is not cfis_globals or cfis_globals.get('permutation_importance') is not original_permutation:
        return False,'Namespace ou permutação original CFIS incompatível.'
    return True,'CFIS validado no namespace atual.'

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

def candidate_external_job(ds,method,k,proto):
    # Esta função é mantida inexecutável até auditoria e liberação em nova versão.
    if not EXTERNAL_TEST_EXECUTION_ALLOWED:
        raise PermissionError('TESTE EXTERNO BLOQUEADO NA FASE 5D. Não modifique o bloqueio; audite antes.')
    # Carregamento dos índices EXTERNOS existentes, sem recriar split.
    source=next((DATA/folder/f'{ds}.parquet' for folder in ('Raw','Synthetic') if (DATA/folder/f'{ds}.parquet').is_file()),None)
    if source is None: raise FileNotFoundError(ds)
    meta=json.loads(source.with_name(f'{ds}_metadata.json').read_text(encoding='utf-8'))
    frame=pd.read_parquet(source)
    train_idx=np.load(DATA/'Splits'/f'{ds}_train_indices.npy',allow_pickle=False)
    test_idx=np.load(DATA/'Splits'/f'{ds}_test_indices.npy',allow_pickle=False)
    if len(set(train_idx.tolist())&set(test_idx.tolist())):raise AssertionError('Sobreposição train/test')
    target=meta['target_column']
    Xtrain=frame.iloc[train_idx].drop(columns=[target]); ytrain=frame.iloc[train_idx][target].to_numpy().ravel()
    Xtest=frame.iloc[test_idx].drop(columns=[target]); ytest=frame.iloc[test_idx][target].to_numpy().ravel()
    if ds=='20_Newsgroups':
        Xtrain=Xtrain['text'].fillna('').astype(str)
        Xtest=Xtest['text'].fillna('').astype(str)
    artifact=joblib.load(DATA/'Processed'/ds/'preprocessor.joblib')
    prep=clone(artifact['preprocessor']); vf=clone(artifact['variance_filter'])
    # Ajuste APENAS em outer train; outer test só recebe transform().
    train_pre=prep.fit_transform(Xtrain,ytrain)
    train_mat=dense_checked(vf.fit_transform(train_pre))
    names=get_names(prep,vf,Xtrain,train_mat.shape[1])
    if not (1<=k<=len(names)):raise AssertionError(f'K inválido: {ds}/{method}/{k} p={len(names)}')
    task='regression' if ds in proto['regression_datasets'] else 'classification'
    start=time.perf_counter()
    ranking=compute_ranking(method,train_mat,ytrain,names,task,0)
    selection_s=time.perf_counter()-start
    if len(ranking)!=len(names) or sorted(map(int,ranking))!=list(range(len(names))):
        raise AssertionError('Ranking inválido')
    chosen=np.asarray(ranking[:k],dtype=int)
    start=time.perf_counter()
    model=tree(task).fit(train_mat[:,chosen],ytrain)
    fit_s=time.perf_counter()-start
    # Transformação de test acontece DEPOIS de todos os fits e da seleção.
    test_mat=dense_checked(vf.transform(prep.transform(Xtest)))
    start=time.perf_counter()
    pred=model.predict(test_mat[:,chosen])
    predict_s=time.perf_counter()-start
    result={'Dataset':ds,'Method':method,'TopK':int(k),'Selection_s':selection_s,'Fit_s':fit_s,
            'Predict_s':predict_s,'Selected_Features':names[chosen].tolist(),'Outer_Test_Used':True}
    if task=='classification':
        result.update(Macro_F1=float(f1_score(ytest,pred,average='macro',zero_division=0)),
                      Balanced_Accuracy=float(balanced_accuracy_score(ytest,pred)))
    else:
        result.update(R2=float(r2_score(ytest,pred)),RMSE=float(np.sqrt(mean_squared_error(ytest,pred))))
    return result

def audit_candidate():
    proto_path=RUN/'Manifests/protocol.json'
    k_path=BASE/'Fase4_Consolidacao_DecisionTree/01_k_congelado_120_combinacoes.csv'
    proto=json.loads(proto_path.read_text(encoding='utf-8'))
    frozen=pd.read_csv(k_path)
    assert proto['model']=='DecisionTree' and proto['seed']==SEED
    assert len(proto['datasets'])==12 and len(proto['methods'])==10
    assert len(frozen)==120 and not frozen[['Dataset','Method']].duplicated().any()
    assert set(frozen.Dataset)==set(proto['datasets']) and set(frozen.Method)==set(proto['methods'])
    assert frozen.K_Status.eq('FROZEN_FROM_INNER_VALIDATION__NO_EXTERNAL_TEST').all()
    assert all(int(r.TopK) in proto['topk'][r.Dataset] for r in frozen.itertuples())
    assert EXTERNAL_TEST_EXECUTION_ALLOWED is False
    # A auditoria NÃO lê Parquet, índices de teste, nem executa candidato.
    cfis_ok,cfis_note=original_cfis_ready()
    OUT.mkdir(parents=True,exist_ok=True)
    plan=frozen[['Dataset','Method','TopK']].copy()
    plan['Status']='BLOCKED_PENDING_FINAL_CODE_AUDIT'
    plan.to_csv(OUT/'01_plano_externo_bloqueado.csv',index=False)
    report={'phase':'5D','status':'CANDIDATE_CREATED_TEST_BLOCKED',
            'protocol_sha256':proto['protocol_sha256'],
            'source_benchmark_sha256':SOURCE_BENCHMARK_SHA256,
            'frozen_k_sha256':file_hash(k_path),'candidate_sha256':file_hash(Path(__file__)) if '__file__' in globals() and Path(__file__).is_file() else None,
            'n_jobs':120,'n_datasets':12,'n_methods':10,
            'cfis_environment_ready':cfis_ok,'cfis_environment_note':cfis_note,
            'external_test_accessed':False,'models_trained':False,
            'open_issues':['Auditar a implementação externa integral antes de liberar.',
                'Definir o tratamento da baseline todas as features e sua métrica de custo.',
                'Confirmar a política de Random_TopK: no original seed=SEED+split_no; aqui split_no=0.',
                'Confirmar disponibilidade do CFIS original validado no notebook.',
                'Avaliar custo de mRMR/RFE em alta dimensionalidade antes de executar.']}
    (OUT/'00_manifesto_candidato.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FASE 5D CONCLUÍDA — CANDIDATO EXTERNO GERADO E BLOQUEADO')
    print('Plano:',len(plan),'combinações | CFIS pronto:',cfis_ok)
    print('Teste externo: NÃO ACESSADO | Modelos: NÃO TREINADOS')
    print('Saída:',OUT)
    print('PRÓXIMO PASSO: auditar candidato e decidir liberação externa.')

SOURCE_BENCHMARK_SHA256='ea709314cde7f7e6ea3692c67ad99cf28bc110c1c74792f4ae524f40e890c666'
if __name__=='__main__':
    audit_candidate()
