# CFIS — FASE 4: fechamento documental do benchmark interno Decision Tree.
# Não treina, não acessa checkpoints nem teste externo, não modifica protocolo original.
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import pandas as pd

RUN=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo/04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1')
SRC=RUN/'Results'/'Auditoria_Cientifica'
OUT=SRC/'Fase4_Consolidacao_DecisionTree'
OUT.mkdir(parents=True,exist_ok=True)
P=RUN/'Manifests'/'protocol.json'
files={
 'pred':SRC/'01_avaliacoes_auditadas.csv',
 'sel':SRC/'02_eficiencia_selecao.csv',
 'summary':SRC/'03_desempenho_por_topk.csv',
 'stab':SRC/'06_estabilidade.csv',
 'agree':SRC/'08_concordancia_resumo.csv',
 'best_old':SRC/'Fase2_Estatistica'/'03_melhor_k_exploratorio.csv',
 'trade':SRC/'Fase3_Tradeoff'/'03_pareto_melhor_k_exploratorio.csv',
 'nogdiag':SRC/'Fase3_Tradeoff'/'05_nogueira_diagnostico.csv',
 'protocol':P,
}
for name,path in files.items():
 if not path.is_file():raise FileNotFoundError(f'Arquivo ausente ({name}): {path}')
print('FASE 4 — Consolidando Decision Tree (somente arquivos derivados)...',flush=True)
D={k:pd.read_csv(v) for k,v in files.items() if k!='protocol'}
proto=json.loads(P.read_text(encoding='utf-8'))
assert len(D['pred'])==8000 and len(D['sel'])==1200 and len(D['stab'])==800
assert D['pred'][['Dataset','Split','Method','TopK']].duplicated().sum()==0
assert D['sel'][['Dataset','Split','Method']].duplicated().sum()==0
assert len(D['pred'].Dataset.unique())==12 and len(D['pred'].Method.unique())==10
assert set(D['pred'].Method)==set(D['sel'].Method)
assert 'CFIS' in set(D['pred'].Method)
assert str(proto.get('model','')).lower() in ('decisiontree','decision_tree','decision tree'), 'Modelo do protocolo não corresponde a Decision Tree'
print('1/4 — Congelando a regra de escolha de K...',flush=True)
# Regra pré-teste externo: maximizar média da métrica primária nos 10 splits internos.
# Desempate: menor K; se necessário, menor tempo médio de seleção.
s=D['summary'].copy()
needed=['Dataset','Method','TopK','Mean_Primary','Mean_Selection_s']
assert all(c in s for c in needed),f'Colunas disponíveis: {list(s.columns)}'
assert s[needed].notna().all().all()
assert s[['Dataset','Method','TopK']].duplicated().sum()==0
s=s.sort_values(['Dataset','Method','Mean_Primary','TopK','Mean_Selection_s'],ascending=[True,True,False,True,True],kind='mergesort')
chosen=s.drop_duplicates(['Dataset','Method']).copy()
assert len(chosen)==120
chosen['Selection_Rule']='maximize_mean_inner_primary_then_min_K_then_min_selection_time'
chosen['K_Status']='FROZEN_FROM_INNER_VALIDATION__NO_EXTERNAL_TEST'
chosen.to_csv(OUT/'01_k_congelado_120_combinacoes.csv',index=False)
# Comparar com a seleção exploratória já publicada, para impedir alteração silenciosa.
old=D['best_old'][['Dataset','Method','TopK']].rename(columns={'TopK':'Previous_K'})
comp=chosen[['Dataset','Method','TopK']].merge(old,on=['Dataset','Method'],validate='one_to_one')
comp['Matches_Previous_Exploratory_K']=comp.TopK==comp.Previous_K
comp.to_csv(OUT/'02_verificacao_k_anterior.csv',index=False)
print('2/4 — Fechando diagnóstico de estabilidade...',flush=True)
st=D['stab'].copy()
st['Nogueira_Status']=np.where(st.Nogueira.notna(),'CALCULATED',np.where(~st.Feature_Universe_Consistent.astype(bool),'NA_INCONSISTENT_FEATURE_UNIVERSE','NA_OTHER_OR_DEGENERATE'))
st.to_csv(OUT/'03_estabilidade_status_800_linhas.csv',index=False)
st.groupby(['Dataset','Nogueira_Status'],dropna=False).size().rename('N').reset_index().to_csv(OUT/'04_nogueira_status_por_dataset.csv',index=False)
# Resumo por dataset: manter Jaccard; Nogueira somente nos casos matematicamente válidos.
chosen_st=chosen[['Dataset','Method','TopK']].merge(st,on=['Dataset','Method','TopK'],validate='one_to_one')
chosen_st.to_csv(OUT/'05_estabilidade_k_congelado.csv',index=False)
print('3/4 — Consolidando resultados internos e rastreabilidade...',flush=True)
trade=D['trade']
required_trade=['Dataset','Method','Pareto_Selection','Pareto_Total']
assert all(c in trade for c in required_trade)
combined=chosen.merge(trade[required_trade],on=['Dataset','Method'],validate='one_to_one')
combined=combined.merge(chosen_st[['Dataset','Method','Mean_Jaccard','Nogueira','Nogueira_Status']],on=['Dataset','Method'],validate='one_to_one')
combined.to_csv(OUT/'06_tabela_integrada_interna_120_linhas.csv',index=False)
# Manifesto com hash dos insumos para reprodutibilidade, sem modificar arquivos existentes.
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
manifest={'study':'CFIS vs nine comparators — Decision Tree only','scope':'internal benchmark consolidation',
 'datasets':12,'methods':10,'inner_splits':10,'predictions':8000,'rankings':1200,
 'k_selection_rule':'For each dataset and method, choose K maximizing mean inner validation primary metric across ten splits; ties: smaller K, then lower mean selection time.',
 'primary_metrics':'Macro-F1 for classification; R2 for regression',
 'selection_is_exploratory':'The same inner splits estimate scores and select K; resulting selected-K scores are not independent external performance estimates.',
 'statistical_caution':'Inner splits overlap; do not treat as independent replicates. Friedman/Wilcoxon results exploratory.',
 'stability_caution':'Missing Nogueira is not zero; inconsistent feature universes cannot be silently aligned.',
 'external_test':'NOT ACCESSED; final evaluation requires a separately audited outer training/test pipeline.',
 'random_forest':'OUT OF SCOPE',
 'inputs_sha256':{k:sha256(v) for k,v in files.items()},
 'chosen_k_matches_previous':int(comp.Matches_Previous_Exploratory_K.sum()),
 'chosen_k_total':len(comp),
 'nogueira_missing':int(st.Nogueira.isna().sum()),
 'nogueira_status_counts':{str(k):int(v) for k,v in st.Nogueira_Status.value_counts().items()},
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'00_manifesto_consolidacao.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
checklist='''# Decision Tree — checklist antes do teste externo\n\n- [x] 8.000 avaliações internas e 1.200 rankings auditados.\n- [x] Dez métodos e doze datasets; sem Random Forest.\n- [x] K escolhido somente por validação interna e regra registrada.\n- [x] Valores ausentes de Nogueira documentados, sem imputação.\n- [ ] Confirmar, no código original, isolamento do teste externo e pré-processamento ajustado somente no treino externo.\n- [ ] Confirmar reconstrução do seletor no treino externo, sem reutilizar rankings dos splits internos.\n- [ ] Confirmar seeds, métricas, custo de seleção/treino e política de empates no código de avaliação externa.\n- [ ] Rodar avaliação externa uma única vez após auditoria do pipeline.\n- [ ] Produzir tabelas e figuras finais, distinguindo claramente interno de externo.\n\n**Atenção:** esta célula não autoriza por si só a execução externa; é preciso inspecionar o pipeline original.\n'''
(OUT/'07_checklist_pre_teste_externo.md').write_text(checklist,encoding='utf-8')
print('4/4 — Relatório final de consolidação...',flush=True)
print('\n'+'='*78)
print('FASE 4 CONCLUÍDA — BENCHMARK INTERNO DECISION TREE CONSOLIDADO')
print('K congelados:',len(chosen),'| Coincidem com K exploratório anterior:',int(comp.Matches_Previous_Exploratory_K.sum()),'/120')
print('Nogueira ausente:',int(st.Nogueira.isna().sum()))
print(st.Nogueira_Status.value_counts().to_string())
print('CFIS em Pareto seleção:',int(combined.loc[combined.Method=='CFIS','Pareto_Selection'].sum()),'/12')
print('CFIS em Pareto custo total:',int(combined.loc[combined.Method=='CFIS','Pareto_Total'].sum()),'/12')
print('Resultados:',OUT)
print('Nenhum modelo treinado. Nenhum checkpoint ou teste externo acessado.')
print('PRÓXIMO PASSO: auditoria do pipeline de teste externo antes de executá-lo.')
