# CÉLULA 53 — Auditoria e consolidação dos checkpoints (sem reexecutar modelos)
from pathlib import Path
from itertools import combinations
import json, hashlib, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
RUN=ROOT/'04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1'
PROTO=RUN/'Manifests/protocol.json'
MANIFEST=RUN/'Manifests/inner_benchmark_manifest.csv'
CKPT=RUN/'Checkpoints'
OUT=RUN/'Results'/'Auditoria_Cientifica'
OUT.mkdir(parents=True,exist_ok=True)
assert PROTO.exists() and MANIFEST.exists(), 'Manifesto/protocolo não encontrados. Monte o Google Drive.'
protocol=json.loads(PROTO.read_text(encoding='utf-8'))
ph=protocol['protocol_sha256']; methods=list(protocol['methods']); datasets=list(protocol['datasets'])
manifest=pd.read_csv(MANIFEST)
assert protocol['model']=='DecisionTree' and len(methods)==10 and len(datasets)==12
assert len(manifest)==8000 and manifest['Protocol_SHA256'].eq(ph).all()
issues=[]; predictions=[]; rankings=[]; selection=[]

def load_checked(path, kind, ds, sn, method, k=None):
    if not path.exists():
        issues.append(dict(tipo='AUSENTE',arquivo=str(path),detalhe=''))
        return None
    try:
        d=json.loads(path.read_text(encoding='utf-8'))
        if d.get('protocol_sha256')!=ph: raise ValueError('Hash de protocolo divergente')
        if kind=='ranking':
            if d.get('dataset')!=ds or int(d.get('split',-1))!=sn or d.get('method')!=method:
                raise ValueError('Identificação do ranking divergente')
            universe=d['feature_names']; order=d['ranking_names']
            if len(universe)!=len(order) or len(set(universe))!=len(universe) or set(universe)!=set(order):
                raise ValueError('Ranking inválido, duplicado ou incompleto')
            if d.get('outer_test_used') is not False: raise ValueError('Flag de teste externo inválida')
            for fld in ['selection_wall_seconds','selection_cpu_seconds']:
                if not np.isfinite(float(d[fld])) or float(d[fld])<0: raise ValueError(f'Tempo inválido: {fld}')
        else:
            if d.get('Dataset')!=ds or int(d.get('Split',-1))!=sn or d.get('Method')!=method or int(d.get('TopK',-1))!=k:
                raise ValueError('Identificação da predição divergente')
            if d.get('Outer_Test_Used') is not False: raise ValueError('Flag de teste externo inválida')
            if d.get('Model')!='DecisionTree': raise ValueError('Modelo inesperado')
            task='regression' if ds in protocol['regression_datasets'] else 'classification'
            metrics=['R2','RMSE'] if task=='regression' else ['Macro_F1','Balanced_Accuracy']
            for fld in metrics+['Fit_Wall_Seconds','Fit_CPU_Seconds']:
                if fld not in d or not np.isfinite(float(d[fld])): raise ValueError(f'Métrica inválida: {fld}')
            if task=='classification' and any(not (0<=float(d[f])<=1) for f in metrics):
                raise ValueError('Métrica de classificação fora de [0,1]')
            if task=='regression' and float(d['RMSE'])<0: raise ValueError('RMSE negativo')
        return d
    except Exception as e:
        issues.append(dict(tipo='INVALIDO',arquivo=str(path),detalhe=repr(e)))
        return None

for ds in datasets:
    for sn in range(1,int(protocol['inner_splits'])+1):
        folder=CKPT/ds/f'split_{sn:02d}'
        for method in methods:
            rr=load_checked(folder/f'ranking__{method}.json','ranking',ds,sn,method)
            if rr is not None:
                rankings.append(dict(Dataset=ds,Split=sn,Method=method,ranking=rr['ranking_names'],universe=rr['feature_names']))
                selection.append(dict(Dataset=ds,Split=sn,Method=method,
                    Selection_Wall_Seconds=float(rr['selection_wall_seconds']),
                    Selection_CPU_Seconds=float(rr['selection_cpu_seconds']),
                    Selection_Peak_RSS_MiB=rr.get('selection_sampled_peak_rss_mib'),
                    Energy_Wh=rr.get('estimated_energy_wh')))
            for k in protocol['topk'][ds]:
                pp=load_checked(folder/f'prediction__{method}__K{k}.json','prediction',ds,sn,method,k)
                if pp is not None:
                    if rr is None: issues.append(dict(tipo='ORFAO',arquivo=str(folder/f'prediction__{method}__K{k}.json'),detalhe='Predição sem ranking válido'))
                    elif int(k)>len(rr['ranking_names']): issues.append(dict(tipo='K_INVALIDO',arquivo=str(folder/f'prediction__{method}__K{k}.json'),detalhe='K maior que universo'))
                    predictions.append(pp)

pred=pd.DataFrame(predictions); sel=pd.DataFrame(selection)
rank_map={(r['Dataset'],r['Split'],r['Method']):r for r in rankings}
if len(pred):
    pred['Primary_Score']=pred.apply(lambda r: float(r['R2']) if r['Dataset'] in protocol['regression_datasets'] else float(r['Macro_F1']),axis=1)
    pred['Primary_Metric']=pred['Dataset'].map(lambda d:'R2' if d in protocol['regression_datasets'] else 'Macro_F1')
    pred.to_csv(OUT/'01_avaliacoes_auditadas.csv',index=False)
if len(sel): sel.to_csv(OUT/'02_eficiencia_selecao.csv',index=False)

# Consolidação com custo de seleção contabilizado uma vez por ranking, e não repetido para cada K.
if len(pred) and len(sel):
    merged=pred.merge(sel,on=['Dataset','Split','Method'],how='left',validate='many_to_one')
    merged['Total_Wall_Seconds']=merged['Selection_Wall_Seconds']+merged['Fit_Wall_Seconds']
    merged['Total_CPU_Seconds']=merged['Selection_CPU_Seconds']+merged['Fit_CPU_Seconds']
    agg={'Mean_Primary':('Primary_Score','mean'),'Std_Primary':('Primary_Score','std'),
         'N_Splits':('Split','nunique'),'Mean_Selection_s':('Selection_Wall_Seconds','mean'),
         'Mean_Fit_s':('Fit_Wall_Seconds','mean'),'Mean_Total_s':('Total_Wall_Seconds','mean')}
    for col in ['Macro_F1','Balanced_Accuracy','R2','RMSE']:
        if col in merged: agg['Mean_'+col]=(col,'mean')
    summary=merged.groupby(['Dataset','Method','TopK'],as_index=False).agg(**agg)
    summary.to_csv(OUT/'03_desempenho_por_topk.csv',index=False)
    # Diferenças pareadas, preservando a mesma divisão e K.
    paired=merged.pivot_table(index=['Dataset','Split','TopK'],columns='Method',values='Primary_Score',aggfunc='first')
    delta=paired.sub(paired['CFIS'],axis=0).mul(-1) if 'CFIS' in paired else pd.DataFrame()
    if len(delta):
        delta=delta.drop(columns=['CFIS'],errors='ignore').stack().rename('Delta_CFIS_menos_Metodo').reset_index().rename(columns={'Method':'Competitor'})
        delta.to_csv(OUT/'04_diferencas_pareadas.csv',index=False)
        delta.groupby(['Dataset','TopK','Competitor'],as_index=False).agg(Mean_Delta=('Delta_CFIS_menos_Metodo','mean'),Std_Delta=('Delta_CFIS_menos_Metodo','std'),N=('Delta_CFIS_menos_Metodo','size')).to_csv(OUT/'05_resumo_diferencas.csv',index=False)
else: summary=pd.DataFrame()

stability=[]; concordance=[]
for ds in datasets:
    ks=protocol['topk'][ds]
    for method in methods:
        recs=[rank_map[(ds,s,method)] for s in range(1,11) if (ds,s,method) in rank_map]
        if len(recs)<2: continue
        universes=[set(r['universe']) for r in recs]
        consistent=all(u==universes[0] for u in universes)
        p=len(universes[0]) if consistent else None
        for k in ks:
            sets=[set(r['ranking'][:k]) for r in recs]
            j=float(np.mean([len(a&b)/len(a|b) for a,b in combinations(sets,2)]))
            nog=np.nan
            if consistent and 0<k<p:
                n=len(sets); freq=np.array([sum(v in s for s in sets)/n for v in sorted(universes[0])])
                variance=(n/(n-1))*np.mean(freq*(1-freq))
                nog=float(1-variance/((k/p)*(1-k/p)))
            stability.append(dict(Dataset=ds,Method=method,TopK=k,Splits=len(recs),Mean_Jaccard=j,Nogueira=nog,Feature_Universe_Consistent=consistent))
    for sn in range(1,11):
        a=rank_map.get((ds,sn,'CFIS'))
        if a is None: continue
        for competitor in methods:
            if competitor=='CFIS': continue
            b=rank_map.get((ds,sn,competitor))
            if b is None: continue
            ra=a['ranking']; rb=b['ranking']
            same=set(ra)==set(rb)
            rho=tau=np.nan
            if same:
                pa={v:i for i,v in enumerate(ra)}; pb={v:i for i,v in enumerate(rb)}
                keys=sorted(pa)
                rho=float(spearmanr([pa[v] for v in keys],[pb[v] for v in keys]).statistic)
                tau=float(kendalltau([pa[v] for v in keys],[pb[v] for v in keys]).statistic)
            for k in ks:
                A=set(ra[:k]);B=set(rb[:k]);inter=len(A&B)
                concordance.append(dict(Dataset=ds,Split=sn,Competitor=competitor,TopK=k,
                    Jaccard=inter/len(A|B),Precision_at_K=inter/k,Spearman=rho,Kendall=tau,
                    Same_Feature_Universe=same))

stab=pd.DataFrame(stability); conc=pd.DataFrame(concordance)
if len(stab): stab.to_csv(OUT/'06_estabilidade.csv',index=False)
if len(conc):
    conc.to_csv(OUT/'07_concordancia_cfis.csv',index=False)
    conc.groupby(['Dataset','Competitor','TopK'],as_index=False).agg(Mean_Jaccard=('Jaccard','mean'),Mean_Precision_at_K=('Precision_at_K','mean'),Mean_Spearman=('Spearman','mean'),Mean_Kendall=('Kendall','mean'),N=('Split','nunique')).to_csv(OUT/'08_concordancia_resumo.csv',index=False)

pd.DataFrame(issues,columns=['tipo','arquivo','detalhe']).to_csv(OUT/'09_problemas.csv',index=False)
report=dict(protocol_sha256=ph,expected_predictions=8000,valid_predictions=len(pred),
            expected_rankings=1200,valid_rankings=len(rankings),issues=len(issues),
            datasets=len(datasets),methods=len(methods),splits=protocol['inner_splits'],
            test_externo_executado_por_esta_celula=False,
            nogueira_na_universo_variavel='Não calculado; Jaccard por nomes calculado',
            energy='Não medida; campos nulos não são zero')
(OUT/'00_relatorio.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*80)
print('AUDITORIA CIENTÍFICA — DECISION TREE (SOMENTE LEITURA DOS CHECKPOINTS)')
print(f'Predições válidas: {len(pred):,}/8.000 | Rankings válidos: {len(rankings):,}/1.200 | Problemas: {len(issues)}')
print(f'Estabilidade: {len(stab)} linhas | Nogueira não calculado: {int(stab.Nogueira.isna().sum()) if len(stab) else 0}')
print(f'Concordância: {len(conc)} linhas | Diretório: {OUT}')
if issues:
    print('\nPRIMEIROS PROBLEMAS:')
    print(pd.DataFrame(issues).head(12).to_string(index=False))
if len(summary):
    print('\nMELHOR K POR DATASET E MÉTODO (seleção exploratória interna; NÃO teste externo):')
    best=summary.sort_values(['Dataset','Method','Mean_Primary','TopK'],ascending=[True,True,False,True]).drop_duplicates(['Dataset','Method'])
    best.to_csv(OUT/'10_melhor_k_interno_por_metodo.csv',index=False)
    print(best[best.Method.isin(['CFIS','mRMR','Mutual_Information','TreeSHAP'])][['Dataset','Method','TopK','Mean_Primary','Std_Primary','Mean_Selection_s']].round(4).to_string(index=False))
print('\nAUDITORIA CONCLUÍDA. Nenhum modelo foi treinado nesta célula.')
