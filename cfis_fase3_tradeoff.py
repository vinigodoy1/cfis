# FASE 3 — Desempenho, estabilidade e eficiência (somente CSVs auditados)
# Sem treinamento, sem checkpoints, sem teste externo, sem alteração do protocolo.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN = Path('/content/drive/MyDrive/CFIS_Artigo_Diogo/04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1')
SRC = RUN/'Results'/'Auditoria_Cientifica'
PH2 = SRC/'Fase2_Estatistica'
OUT = SRC/'Fase3_Tradeoff'
OUT.mkdir(parents=True, exist_ok=True)
needed = [SRC/'01_avaliacoes_auditadas.csv',SRC/'02_eficiencia_selecao.csv',SRC/'03_desempenho_por_topk.csv',SRC/'06_estabilidade.csv',PH2/'03_melhor_k_exploratorio.csv']
for p in needed:
    if not p.is_file(): raise FileNotFoundError(f'Arquivo necessário não encontrado: {p}')
print('FASE 3: lendo os resultados consolidados (sem acessar checkpoints)...',flush=True)
pred = pd.read_csv(needed[0]); sel = pd.read_csv(needed[1]); summ = pd.read_csv(needed[2]); stab = pd.read_csv(needed[3]); best = pd.read_csv(needed[4])
assert len(pred)==8000 and len(sel)==1200 and len(stab)==800, 'Contagens incompatíveis com a auditoria.'
assert pred[['Dataset','Split','Method','TopK']].duplicated().sum()==0
assert sel[['Dataset','Split','Method']].duplicated().sum()==0
assert best[['Dataset','Method']].duplicated().sum()==0
assert len(best)==120
regression = {'California_Housing','Diabetes_Reg'}
methods = sorted(pred.Method.unique()); datasets = sorted(pred.Dataset.unique())
assert len(methods)==10 and len(datasets)==12 and 'CFIS' in methods

print('1/5 — Mesmo K: desempenho e razão de tempo de seleção...',flush=True)
# Um tempo de seleção por split e método (ranking reutilizado para diferentes K).
s = sel[['Dataset','Split','Method','Selection_Wall_Seconds']].copy()
assert (pd.to_numeric(s.Selection_Wall_Seconds,errors='coerce')>=0).all()
p = pred[['Dataset','Split','Method','TopK','Macro_F1','R2']].copy()
p['Score'] = np.where(p.Dataset.isin(regression),pd.to_numeric(p.R2,errors='coerce'),pd.to_numeric(p.Macro_F1,errors='coerce'))
assert np.isfinite(p.Score).all()
p = p.merge(s,on=['Dataset','Split','Method'],validate='many_to_one')
cf = p[p.Method=='CFIS'][['Dataset','Split','TopK','Score','Selection_Wall_Seconds']].rename(columns={'Score':'CFIS_Score','Selection_Wall_Seconds':'CFIS_Selection_s'})
cp = p[p.Method!='CFIS'].rename(columns={'Method':'Competitor','Score':'Competitor_Score','Selection_Wall_Seconds':'Competitor_Selection_s'})
cp = cp[['Dataset','Split','TopK','Competitor','Competitor_Score','Competitor_Selection_s']].merge(cf,on=['Dataset','Split','TopK'],validate='many_to_one')
cp['Delta_Score'] = cp.CFIS_Score-cp.Competitor_Score
cp['Speedup_CFIS'] = np.where(cp.CFIS_Selection_s>0,cp.Competitor_Selection_s/cp.CFIS_Selection_s,np.nan)
cp['CFIS_Faster'] = cp.CFIS_Selection_s<cp.Competitor_Selection_s
cp['CFIS_Better_Score'] = cp.Delta_Score>0
cp.to_csv(OUT/'01_comparacoes_pareadas_mesmo_k.csv',index=False)
agg=cp.groupby(['Dataset','TopK','Competitor'],as_index=False).agg(
    Mean_Delta_Score=('Delta_Score','mean'),Median_Delta_Score=('Delta_Score','median'),
    CFIS_Wins_Score=('CFIS_Better_Score','sum'),CFIS_Faster_Splits=('CFIS_Faster','sum'),
    Mean_CFIS_Selection_s=('CFIS_Selection_s','mean'),Mean_Competitor_Selection_s=('Competitor_Selection_s','mean'),N=('Split','size'))
agg['Speedup_Ratio_of_Means'] = np.where(agg.Mean_CFIS_Selection_s>0,agg.Mean_Competitor_Selection_s/agg.Mean_CFIS_Selection_s,np.nan)
agg.to_csv(OUT/'02_comparacao_mesmo_k_resumo.csv',index=False)

print('2/5 — Fronteiras de Pareto por dataset (K exploratório)...',flush=True)
trade=best[['Dataset','Method','TopK','Mean_Primary','Mean_Selection_s','Mean_Fit_s','Mean_Total_s']].copy()
assert np.isfinite(trade[['Mean_Primary','Mean_Selection_s','Mean_Total_s']].to_numpy(float)).all()
assert (trade[['Mean_Selection_s','Mean_Total_s']]>=0).all().all()
def pareto_flags(df, time_col):
    score=df.Mean_Primary.to_numpy(float); time=df[time_col].to_numpy(float)
    return [not np.any((score>=score[i]) & (time<=time[i]) & ((score>score[i]) | (time<time[i]))) for i in range(len(df))]
chunks=[]
for ds,g in trade.groupby('Dataset',sort=True):
    g=g.copy()
    g['Pareto_Selection']=pareto_flags(g,'Mean_Selection_s')
    g['Pareto_Total']=pareto_flags(g,'Mean_Total_s')
    chunks.append(g)
trade=pd.concat(chunks,ignore_index=True)
trade.to_csv(OUT/'03_pareto_melhor_k_exploratorio.csv',index=False)

print('3/5 — Estabilidade: mesma dimensão e valores ausentes...',flush=True)
# Comparação de Jaccard no mesmo dataset/K; Nogueira só onde ambas as estimativas existem.
cfstab=stab[stab.Method=='CFIS'][['Dataset','TopK','Mean_Jaccard','Nogueira','Feature_Universe_Consistent']].rename(columns={'Mean_Jaccard':'CFIS_Jaccard','Nogueira':'CFIS_Nogueira','Feature_Universe_Consistent':'CFIS_Universe_Consistent'})
othstab=stab[stab.Method!='CFIS'].rename(columns={'Method':'Competitor','Mean_Jaccard':'Competitor_Jaccard','Nogueira':'Competitor_Nogueira','Feature_Universe_Consistent':'Competitor_Universe_Consistent'})
st=othstab[['Dataset','TopK','Competitor','Competitor_Jaccard','Competitor_Nogueira','Competitor_Universe_Consistent']].merge(cfstab,on=['Dataset','TopK'],validate='many_to_one')
st['Delta_Jaccard']=st.CFIS_Jaccard-st.Competitor_Jaccard
st['Delta_Nogueira_If_Available']=st.CFIS_Nogueira-st.Competitor_Nogueira
st['Nogueira_Both_Available']=st.CFIS_Nogueira.notna()&st.Competitor_Nogueira.notna()
st.to_csv(OUT/'04_estabilidade_pareada_mesmo_k.csv',index=False)
miss=stab[stab.Nogueira.isna()].copy()
miss['Possible_Reason']=np.where(~miss.Feature_Universe_Consistent.astype(bool),'Universo de características inconsistente entre splits','Outro caso: verificar K=p, variância ou condições da fórmula')
miss.to_csv(OUT/'05_nogueira_diagnostico.csv',index=False)
miss.groupby(['Dataset','Possible_Reason'],as_index=False).size().rename(columns={'size':'N_Missing'}).to_csv(OUT/'06_nogueira_ausencias_por_dataset.csv',index=False)

print('4/5 — Resumo integrado e indicadores de Pareto...',flush=True)
# Agregação por dataset evita comparar diretamente R² com Macro-F1.
perds=trade.groupby('Dataset').agg(Methods_Pareto_Selection=('Pareto_Selection','sum'),Methods_Pareto_Total=('Pareto_Total','sum')).reset_index()
cftrade=trade[trade.Method=='CFIS'].copy()
cftrade['CFIS_Pareto_Selection']=cftrade.Pareto_Selection
cftrade['CFIS_Pareto_Total']=cftrade.Pareto_Total
cftrade=cftrade[['Dataset','TopK','Mean_Primary','Mean_Selection_s','Mean_Total_s','CFIS_Pareto_Selection','CFIS_Pareto_Total']]
cftrade.to_csv(OUT/'07_cfis_por_dataset.csv',index=False)
summary=trade.groupby('Method',as_index=False).agg(Pareto_Selection_Datasets=('Pareto_Selection','sum'),Pareto_Total_Datasets=('Pareto_Total','sum'),Median_Selection_s=('Mean_Selection_s','median'),Median_Total_s=('Mean_Total_s','median'))
summary['Pareto_Selection_Fraction']=summary.Pareto_Selection_Datasets/len(datasets)
summary.to_csv(OUT/'08_resumo_pareto_por_metodo.csv',index=False)
# Eficiência relativa por dataset: razão de médias de tempo (não média das razões individuais).
timewide=trade.pivot(index='Dataset',columns='Method',values='Mean_Selection_s')
rel=[]
for ds in datasets:
    for m in methods:
        if m=='CFIS': continue
        tcf=float(timewide.loc[ds,'CFIS']); tm=float(timewide.loc[ds,m]);
        rel.append({'Dataset':ds,'Competitor':m,'CFIS_Selection_s':tcf,'Competitor_Selection_s':tm,'Speedup_CFIS':tm/tcf if tcf>0 else np.nan,'CFIS_Faster':tcf<tm})
rel=pd.DataFrame(rel)
rel.to_csv(OUT/'09_eficiencia_relativa_por_dataset.csv',index=False)
# Não confundir desempenho similar com equivalência formal: limiar somente descritivo.
compare=trade[['Dataset','Method','Mean_Primary','Mean_Selection_s']].merge(
    trade[trade.Method=='CFIS'][['Dataset','Mean_Primary','Mean_Selection_s']].rename(columns={'Mean_Primary':'CFIS_Primary','Mean_Selection_s':'CFIS_Time'}),on='Dataset',validate='many_to_one')
compare=compare[compare.Method!='CFIS'].copy()
compare['CFIS_Minus_Competitor_Score']=compare.CFIS_Primary-compare.Mean_Primary
compare['CFIS_Faster']=compare.CFIS_Time<compare.Mean_Selection_s
compare.to_csv(OUT/'10_tradeoff_comparacao_melhor_k.csv',index=False)

print('5/5 — Gráficos e relatório...',flush=True)
plt.figure(figsize=(10,5)); q=summary.sort_values('Pareto_Selection_Datasets');plt.barh(q.Method,q.Pareto_Selection_Datasets);plt.xlabel('Datasets em que o método pertence à fronteira de Pareto');plt.title('Pareto: desempenho × tempo de seleção (K exploratório)');plt.tight_layout();plt.savefig(OUT/'11_pareto_frequencia.png',dpi=170);plt.close()
for ds in datasets:
    d=trade[trade.Dataset==ds]
    plt.figure(figsize=(9,5.5))
    for _,r in d.iterrows():
        mark='*' if r.Pareto_Selection else 'o'
        plt.scatter(r.Mean_Selection_s,r.Mean_Primary,marker=mark,s=100 if r.Method=='CFIS' else 42)
        plt.annotate(r.Method,(r.Mean_Selection_s,r.Mean_Primary),fontsize=7,xytext=(3,3),textcoords='offset points')
    plt.xscale('symlog',linthresh=.05);plt.xlabel('Tempo médio de seleção (s; escala symlog)')
    plt.ylabel('R²' if ds in regression else 'Macro-F1')
    plt.title(f'{ds} — Pareto (estrela: não dominado; K exploratório)')
    plt.tight_layout();plt.savefig(OUT/f'12_pareto_{ds}.png',dpi=155);plt.close()
report={'datasets':len(datasets),'methods':len(methods),'predictions':len(pred),'rankings':len(sel),
        'same_k_pairs':len(cp),'stability_pairs':len(st),'nogueira_missing':len(miss),
        'nogueira_missing_by_reason':{str(k):int(v) for k,v in miss.Possible_Reason.value_counts().items()},
        'cfis_pareto_selection_datasets':int(trade.loc[trade.Method=='CFIS','Pareto_Selection'].sum()),
        'cfis_pareto_total_datasets':int(trade.loc[trade.Method=='CFIS','Pareto_Total'].sum()),
        'caveats':['Melhor K por método foi escolhido nos mesmos splits internos: análise exploratória, sujeita a otimismo',
                   'Fronteira de Pareto não demonstra significância estatística nem equivalência preditiva',
                   'Não agregar R² e Macro-F1 como valores diretamente comparáveis',
                   'Tempo de seleção por split é reutilizado entre K; não somar ou duplicar custo por K',
                   'Nogueira ausente não é zero; manter ausente e diagnosticar universo de features',
                   'Não houve treinamento, leitura de checkpoints ou uso de teste externo']}
(OUT/'00_relatorio_fase3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n'+'='*76)
print('FASE 3 CONCLUÍDA — DESEMPENHO × ESTABILIDADE × EFICIÊNCIA')
print(f'Predições: {len(pred)} | Rankings: {len(sel)} | Pares no mesmo K: {len(cp)}')
print(f'CFIS na fronteira Pareto (seleção): {report["cfis_pareto_selection_datasets"]}/12 datasets')
print(f'CFIS na fronteira Pareto (custo total): {report["cfis_pareto_total_datasets"]}/12 datasets')
print(f'Nogueira ausente: {len(miss)}')
print('\nFREQUÊNCIA DE PARETO POR MÉTODO:')
print(summary[['Method','Pareto_Selection_Datasets','Pareto_Total_Datasets']].sort_values('Pareto_Selection_Datasets',ascending=False).to_string(index=False))
print('\nDIAGNÓSTICO NOGUEIRA:')
print(miss.Possible_Reason.value_counts().to_string())
print('\nRESULTADOS:',OUT)
print('Nenhum modelo treinado; teste externo não acessado.')
