# Executar no Colab com Google Drive montado. Não treina e não altera resultados históricos.
from pathlib import Path
from itertools import combinations
import json, shutil
import numpy as np
import pandas as pd

RUN = Path('/content/drive/MyDrive/CFIS_Artigo_Diogo/04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1')
CKPT = RUN/'Checkpoints'
PROTO = RUN/'Manifests/protocol.json'
BASE = RUN/'Results/Benchmark_Final_Classification_10/Fechamento_Artigo'
OUT = BASE/'Estabilidade_Recalculada_10_Splits'
OUT.mkdir(parents=True, exist_ok=True)
proto = json.loads(PROTO.read_text(encoding='utf-8'))
methods = list(proto['methods'])
datasets = [d for d in proto['datasets'] if d not in set(proto['regression_datasets'])]
n_splits = int(proto['inner_splits'])
assert n_splits == 10 and len(datasets)==10 and len(methods)==10
ph = proto['protocol_sha256']
records={}
for ds in datasets:
    for method in methods:
        rec=[]
        for s in range(1, n_splits+1):
            path=CKPT/ds/f'split_{s:02d}'/f'ranking__{method}.json'
            if not path.exists(): raise FileNotFoundError(path)
            r=json.loads(path.read_text(encoding='utf-8'))
            assert r['protocol_sha256']==ph and r['dataset']==ds and r['method']==method
            assert int(r['split'])==s and r['outer_test_used'] is False
            names=list(r['feature_names']); order=list(r['ranking_names'])
            assert len(names)==len(set(names))==len(order) and set(names)==set(order)
            rec.append((s, names, order))
        records[ds,method]=rec

# Mesmo K para os dez métodos em cada dataset. Prioriza K próximo de 50% das features.
# Usa somente K que já fazia parte da grade histórica.
chosen_k={}
for ds in datasets:
    common=set(map(int,proto['topk'][ds]))
    assert common
    p_min=min(len(r[1]) for m in methods for r in records[ds,m])
    eligible=[k for k in common if 0<k<p_min]
    if not eligible: raise ValueError(f'{ds}: nenhum K válido comum')
    chosen_k[ds]=min(eligible,key=lambda k:(abs(k/p_min-0.5),k))

rng=np.random.default_rng(20261029)
detail=[]; summary=[]; flags=[]
for ds in datasets:
    k=chosen_k[ds]
    for method in methods:
        rec=records[ds,method]
        universes=[set(x[1]) for x in rec]
        consistent=all(u==universes[0] for u in universes)
        sets=[set(x[2][:k]) for x in rec]
        vals=[]
        for (i,a),(j,b) in combinations(enumerate(sets,1),2):
            score=len(a&b)/len(a|b)
            vals.append(score)
            detail.append(dict(Dataset=ds,Method=method,TopK=k,Split_A=i,Split_B=j,Jaccard=score,Universe_Consistent=consistent))
        vals=np.array(vals,dtype=float)
        # Bootstrap dos splits (não dos 45 pares, que não são independentes).
        boots=[]
        for _ in range(2000):
            ids=rng.integers(0,n_splits,n_splits)
            v=[len(sets[ids[i]]&sets[ids[j]])/len(sets[ids[i]]|sets[ids[j]]) for i,j in combinations(range(n_splits),2)]
            boots.append(np.mean(v))
        ci=np.quantile(boots,[.025,.975])
        nog=np.nan
        status='VALIDO'
        if not consistent:
            status='UNIVERSO_INCONSISTENTE'
        else:
            p=len(universes[0])
            if not (0<k<p):
                status='K_DEGENERADO'
            else:
                counts=np.array([sum(f in s for s in sets) for f in sorted(universes[0])],dtype=float)
                freq=counts/n_splits
                variance=(n_splits/(n_splits-1))*np.mean(freq*(1-freq))
                nog=1-variance/((k/p)*(1-k/p))
        summary.append(dict(Dataset=ds,Method=method,TopK=k,N_Splits=n_splits,N_Pairs=45,
            Mean_Jaccard=float(vals.mean()),SD_Pairwise_Jaccard=float(vals.std(ddof=1)),
            Bootstrap_CI95_Low=float(ci[0]),Bootstrap_CI95_High=float(ci[1]),
            Nogueira=nog,Nogueira_Status=status,Feature_Universe_Consistent=consistent))
        if status!='VALIDO': flags.append(dict(Dataset=ds,Method=method,Issue=status))

detail=pd.DataFrame(detail); summary=pd.DataFrame(summary); flags=pd.DataFrame(flags,columns=['Dataset','Method','Issue'])
assert len(detail)==4500 and len(summary)==100
aggregate=summary.groupby('Method',as_index=False).agg(
    Datasets=('Dataset','nunique'),Mean_Jaccard=('Mean_Jaccard','mean'),
    SD_Jaccard_Across_Datasets=('Mean_Jaccard','std'),
    Mean_Nogueira_Valid=('Nogueira','mean'),N_Nogueira_Valid=('Nogueira','count'))
aggregate=aggregate.sort_values('Mean_Jaccard',ascending=False)
coverage=pd.DataFrame([{'Dataset':d,'Common_K':chosen_k[d]} for d in datasets])
for name,frame in [('01_45_Pares_Por_Dataset_Metodo.csv',detail),('02_Estabilidade_Mesmo_K_100.csv',summary),
                   ('03_Resumo_10_Metodos.csv',aggregate),('04_K_Usado_Por_Dataset.csv',coverage),('05_Alertas.csv',flags)]:
    frame.to_csv(OUT/name,index=False)

# Não sobrescrever a planilha original. Gerar uma cópia atualizada, preservando as abas existentes.
original=BASE/'CFIS_Tabelas_Finais_Artigo.xlsx'
updated=OUT/'CFIS_Tabelas_Finais_Artigo_COM_Estabilidade_Real.xlsx'
if not original.exists(): raise FileNotFoundError(original)
shutil.copy2(original,updated)
with pd.ExcelWriter(updated,engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
    summary.to_excel(writer,sheet_name='Estab_Real_Mesmo_K',index=False)
    aggregate.to_excel(writer,sheet_name='Estab_Real_Resumo',index=False)
    coverage.to_excel(writer,sheet_name='Estab_K_Comparavel',index=False)
    flags.to_excel(writer,sheet_name='Estab_Alertas',index=False)

print('\nESTABILIDADE REAL RECOMPUTADA DOS RANKINGS HISTÓRICOS')
print('10 datasets | 10 métodos | 10 splits | 45 pares por combinação')
print('4500 pares de splits | 100 resultados dataset-método')
print('Nogueira válido:',summary.Nogueira.notna().sum(),'/100')
print('\nK comum por dataset:\n',coverage.to_string(index=False))
print('\nRanking descritivo Jaccard:\n',aggregate.to_string(index=False))
print('\nNova planilha:',updated)
print('Original preservado:',original)
