# FASE 2 — Análise estatística do benchmark CFIS (somente CSVs auditados)
# Não lê checkpoints, não ajusta modelos e não acessa o teste externo.
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

RUN=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo/04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1')
SRC=RUN/'Results'/'Auditoria_Cientifica'
OUT=SRC/'Fase2_Estatistica'
OUT.mkdir(parents=True,exist_ok=True)
for f in ['01_avaliacoes_auditadas.csv','02_eficiencia_selecao.csv','03_desempenho_por_topk.csv','06_estabilidade.csv','07_concordancia_cfis.csv']:
    if not (SRC/f).exists(): raise FileNotFoundError(f'Falta {SRC/f}. Execute a auditoria primeiro.')
print('FASE 2: lendo os CSVs auditados...',flush=True)
pred=pd.read_csv(SRC/'01_avaliacoes_auditadas.csv')
sel=pd.read_csv(SRC/'02_eficiencia_selecao.csv')
summ=pd.read_csv(SRC/'03_desempenho_por_topk.csv')
stab=pd.read_csv(SRC/'06_estabilidade.csv')
conc=pd.read_csv(SRC/'07_concordancia_cfis.csv')
assert len(pred)==8000 and len(sel)==1200, 'Contagens inesperadas: confira a auditoria.'
methods=sorted(pred.Method.unique()); datasets=sorted(pred.Dataset.unique())
assert len(methods)==10 and len(datasets)==12 and 'CFIS' in methods
assert not pred.duplicated(['Dataset','Split','Method','TopK']).any()
assert not sel.duplicated(['Dataset','Split','Method']).any()
regression={'California_Housing','Diabetes_Reg'}
# Cada dataset é comparado apenas consigo mesmo, usando a métrica primária correspondente.
pred['Primary_Score']=np.where(pred.Dataset.isin(regression),pd.to_numeric(pred.R2,errors='coerce'),pd.to_numeric(pred.Macro_F1,errors='coerce'))
assert np.isfinite(pred.Primary_Score).all(), 'Há métricas primárias inválidas.'

print('1/5 — Diferenças pareadas CFIS × 9 métodos (mesmo dataset, split e K)...',flush=True)
base=pred[pred.Method=='CFIS'][['Dataset','Split','TopK','Primary_Score']].rename(columns={'Primary_Score':'CFIS_Score'})
other=pred[pred.Method!='CFIS'][['Dataset','Split','TopK','Method','Primary_Score']].rename(columns={'Method':'Competitor','Primary_Score':'Competitor_Score'})
pairs=other.merge(base,on=['Dataset','Split','TopK'],how='inner',validate='many_to_one')
pairs['Delta_CFIS']=pairs.CFIS_Score-pairs.Competitor_Score
pairs.to_csv(OUT/'01_pares_cfis.csv',index=False)
rows=[]
for (ds,k),sub in pairs.groupby(['Dataset','TopK'],sort=True):
    group=[]
    for competitor,g in sub.groupby('Competitor',sort=True):
        d=g.sort_values('Split').Delta_CFIS.to_numpy(float)
        n=len(d); avg=float(np.mean(d)); sd=float(np.std(d,ddof=1)) if n>1 else np.nan
        # Descritivo, não intervalo de confiança independente: splits se sobrepõem.
        try:
            p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue) if np.any(d!=0) else 1.0
        except ValueError: p=np.nan
        group.append(dict(Dataset=ds,TopK=int(k),Competitor=competitor,N=n,
                          Mean_Delta=avg,SD_Delta=sd,Median_Delta=float(np.median(d)),
                          Win_Splits=int(np.sum(d>0)),Tie_Splits=int(np.sum(d==0)),Loss_Splits=int(np.sum(d<0)),
                          Wilcoxon_p_exploratorio=p))
    ps=np.array([r['Wilcoxon_p_exploratorio'] for r in group],float)
    valid=np.isfinite(ps); adjusted=np.full(len(ps),np.nan)
    if valid.any(): adjusted[valid]=multipletests(ps[valid],method='holm')[1]
    for r,padj in zip(group,adjusted):
        r['Holm_p_exploratorio']=padj
        r['Aviso']='Splits sobrepostos: p exploratório, não evidência confirmatória independente'
        rows.append(r)
pairsummary=pd.DataFrame(rows)
pairsummary.to_csv(OUT/'02_wilcoxon_holm_por_dataset_k.csv',index=False)

print('2/5 — Comparação global por dataset (rankings, sem misturar escalas de métricas)...',flush=True)
# Escolha exploratória do K de cada método baseada nas médias internas.
best=summ.sort_values(['Dataset','Method','Mean_Primary','TopK'],ascending=[True,True,False,True]).drop_duplicates(['Dataset','Method']).copy()
best.to_csv(OUT/'03_melhor_k_exploratorio.csv',index=False)
best['Rank_Dataset']=best.groupby('Dataset').Mean_Primary.rank(ascending=False,method='average')
best[['Dataset','Method','TopK','Mean_Primary','Rank_Dataset','Mean_Selection_s','Mean_Total_s']].to_csv(OUT/'04_ranking_por_dataset.csv',index=False)
wide=best.pivot(index='Dataset',columns='Method',values='Rank_Dataset').reindex(columns=methods)
assert wide.notna().all().all()
stat,p_global=friedmanchisquare(*[wide[c].to_numpy() for c in methods])
meanranks=wide.mean().sort_values().rename('Mean_Rank').reset_index()
meanranks.columns=['Method','Mean_Rank']
meanranks.to_csv(OUT/'05_ranking_medio.csv',index=False)
# Comparação dataset-a-dataset do posto do CFIS, Holm em nove concorrentes.
post=[]
for m in methods:
    if m=='CFIS': continue
    diff=(wide['CFIS']-wide[m]).to_numpy(float) # negativo favorece CFIS
    pv=float(wilcoxon(diff,alternative='two-sided').pvalue) if np.any(diff!=0) else 1.0
    post.append(dict(Competitor=m,Mean_Rank_Diff_CFIS_menos_Competitor=float(diff.mean()),
                     CFIS_Better_Datasets=int((diff<0).sum()),CFIS_Worse_Datasets=int((diff>0).sum()),
                     Tied_Datasets=int((diff==0).sum()),Wilcoxon_p=pv))
post=pd.DataFrame(post)
post['Holm_p']=multipletests(post.Wilcoxon_p.to_numpy(),method='holm')[1]
post.to_csv(OUT/'06_comparacao_global_pareada.csv',index=False)

print('3/5 — Estabilidade e concordância...',flush=True)
stab.to_csv(OUT/'07_estabilidade_detalhada.csv',index=False)
stab.groupby('Method',as_index=False).agg(Jaccard_Medio=('Mean_Jaccard','mean'),Nogueira_Medio_Disponivel=('Nogueira','mean'),Nogueira_Nao_Calculado=('Nogueira',lambda x:int(x.isna().sum())),Linhas=('TopK','size')).to_csv(OUT/'08_estabilidade_por_metodo.csv',index=False)
missing=stab[stab.Nogueira.isna()].copy()
missing.to_csv(OUT/'09_diagnostico_nogueira_ausente.csv',index=False)
conc.groupby('Competitor',as_index=False).agg(Jaccard_Medio=('Jaccard','mean'),Precision_at_K_Media=('Precision_at_K','mean'),Spearman_Medio=('Spearman','mean'),Kendall_Medio=('Kendall','mean'),Linhas=('TopK','size')).to_csv(OUT/'10_concordancia_resumo_global.csv',index=False)

print('4/5 — Eficiência: tempo de seleção e custo total...',flush=True)
# Média de tempo de seleção por dataset e método; não duplicar custo do ranking pelos K.
sel.groupby(['Dataset','Method'],as_index=False).agg(Selection_s=('Selection_Wall_Seconds','mean'),CPU_s=('Selection_CPU_Seconds','mean')).to_csv(OUT/'11_tempo_selecao_por_dataset.csv',index=False)
# Tempo total por K: seleção + treino; melhor K é exploratório.
trade=best[['Dataset','Method','TopK','Mean_Primary','Mean_Selection_s','Mean_Fit_s','Mean_Total_s']].copy()
trade.to_csv(OUT/'12_desempenho_versus_custo.csv',index=False)

print('5/5 — Gerando gráficos e relatório...',flush=True)
plt.figure(figsize=(10,5))
plot=meanranks.sort_values('Mean_Rank',ascending=False)
plt.barh(plot.Method,plot.Mean_Rank)
plt.xlabel('Posto médio (menor é melhor)')
plt.title('Decision Tree — ranking médio em 12 datasets\nK exploratório selecionado por método')
plt.tight_layout();plt.savefig(OUT/'13_ranking_medio.png',dpi=170);plt.close()
for ds in ['20_Newsgroups','Gisette','Synthetic_120','Breast_Cancer']:
    s=trade[trade.Dataset==ds]
    plt.figure(figsize=(8,5))
    for _,r in s.iterrows():
        plt.scatter(r.Mean_Selection_s,r.Mean_Primary,s=45)
        plt.annotate(r.Method,(r.Mean_Selection_s,r.Mean_Primary),fontsize=7,xytext=(3,3),textcoords='offset points')
    plt.xscale('symlog',linthresh=0.1)
    plt.xlabel('Tempo médio de seleção (s; escala simétrica log)')
    plt.ylabel('R²' if ds in regression else 'Macro-F1')
    plt.title(f'{ds}: desempenho × tempo de seleção (K exploratório)')
    plt.tight_layout();plt.savefig(OUT/f'14_pareto_{ds}.png',dpi=170);plt.close()
report={
 'expected_predictions':8000,'actual_predictions':len(pred),
 'expected_rankings':1200,'actual_rankings':len(sel),
 'datasets':len(datasets),'methods':len(methods),
 'friedman_statistic_exploratory_best_k':float(stat),'friedman_p_exploratory_best_k':float(p_global),
 'nogueira_missing':len(missing),
 'warnings':['K escolhido pela mesma validação interna: resultados otimistas, apenas exploratórios',
             'Splits internos compartilham observações; Wilcoxon por split não é teste confirmatório',
             'Friedman sobre postos de datasets, não sobre métricas de escalas diferentes',
             'Não houve acesso ao conjunto de teste externo',
             'Energia não medida; não inferir valores']}
(OUT/'00_relatorio_fase2.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('\n'+'='*76)
print('FASE 2 CONCLUÍDA — ANÁLISE ESTATÍSTICA EXPLORATÓRIA')
print(f'Predições: {len(pred)} | Rankings: {len(sel)} | Datasets: {len(datasets)} | Métodos: {len(methods)}')
print(f'Friedman (postos por dataset, K exploratório): chi²={stat:.4f}, p={p_global:.6g}')
print(f'Nogueira ausente: {len(missing)} linhas')
print('\nRANKING MÉDIO (menor é melhor):')
print(meanranks.round(4).to_string(index=False))
print('\nCFIS × CONCORRENTES — postos por dataset (p corrigido Holm):')
print(post.round(4).to_string(index=False))
print('\nArquivos salvos em:',OUT)
print('Nenhum modelo foi treinado; teste externo não utilizado.')
