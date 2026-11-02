# CFIS — Fase 5C: auditoria estática e plano da execução externa.
# Nenhuma amostra externa é lida; nenhum seletor/modelo é treinado.
from pathlib import Path
import ast, base64, zlib, hashlib, json, datetime
import pandas as pd

ORIGINAL_COMPRESSED = """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"""
ORIGINAL = zlib.decompress(base64.b64decode(ORIGINAL_COMPRESSED)).decode('utf-8')
ROOT=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
RUN=ROOT/'04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1'
BASE=RUN/'Results/Auditoria_Cientifica'
OUT=BASE/'Fase5C_Revisao_Estatica_DT'
OUT.mkdir(parents=True,exist_ok=True)
P=RUN/'Manifests/protocol.json'
K=BASE/'Fase4_Consolidacao_DecisionTree/01_k_congelado_120_combinacoes.csv'
B=BASE/'Fase5B_Auditoria_Pipeline_DT/01_auditoria_12_datasets.csv'
if not B.exists():
    candidates=list((BASE/'Fase5B_Auditoria_Pipeline_DT').glob('*.csv'))
    B=next((p for p in candidates if '12_datasets' in p.name),B)
for p in (P,K):
    if not p.is_file():raise FileNotFoundError(p)
proto=json.loads(P.read_text(encoding='utf-8'))
k=pd.read_csv(K)
assert proto['model']=='DecisionTree' and proto['seed']==42
assert len(proto['datasets'])==12 and len(proto['methods'])==10
assert len(k)==120 and not k[['Dataset','Method']].duplicated().any()
assert set(k.Dataset)==set(proto['datasets']) and set(k.Method)==set(proto['methods'])
assert k.K_Status.eq('FROZEN_FROM_INNER_VALIDATION__NO_EXTERNAL_TEST').all()
assert all(int(r.TopK) in set(map(int,proto['topk'][r.Dataset])) for r in k.itertuples())
if B.is_file():
    b=pd.read_csv(B)
    if 'Status' in b.columns:assert len(b)==12 and b.Status.eq('OK').all()

module=ast.parse(ORIGINAL)
funcs={n.name:n for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
expected=['load_data','get_names','tree','cfis_ranking','compute_ranking','mrmr_ranking','cfs_ranking','l1_ranking','pfi_ranking','shap_ranking','eval_job']
missing=[n for n in expected if n not in funcs]
assert not missing, f'Funções originais ausentes: {missing}'
source_hash=hashlib.sha256(ORIGINAL.encode('utf-8')).hexdigest()
# Conferência da lógica original por AST (sem executar o código).
load=ast.get_source_segment(ORIGINAL,funcs['load_data'])
rank=ast.get_source_segment(ORIGINAL,funcs['compute_ranking'])
cfis=ast.get_source_segment(ORIGINAL,funcs['cfis_ranking'])
checks={
 'Original utiliza indices de treino': '_train_indices.npy' in load,
 'Original não carrega indices de teste': '_test_indices.npy' not in load,
 'Seletores usam conjunto X,y recebido': 'compute_ranking(method,X,y,names,task,split_no)' in ORIGINAL,
 'CFIS original invocado sem modificar arquitetura': 'cfis_function(' in cfis,
 'CFIS original valida adaptador': 'consolidated_permutation_adapter' in cfis,
 'Ranking contém todos os 10 métodos': all(('method=='+repr(m)) in rank for m in proto['methods']),
 'Modelo declarado DecisionTree': 'DecisionTreeClassifier' in ORIGINAL and 'DecisionTreeRegressor' in ORIGINAL,
 'Preprocessamento clonado': "clone(artifact['preprocessor'])" in ORIGINAL and "clone(artifact['variance_filter'])" in ORIGINAL,
 'Preprocessamento interno fit exclusivo no treino': 'prep.fit_transform(xfitraw,yfit)' in ORIGINAL and 'prep.transform(xvalraw)' in ORIGINAL,
 'Teste externo bloqueado no benchmark original': "'Outer_Test_Used':False" in ORIGINAL,
}
rows=[{'Check':a,'OK':bool(v)} for a,v in checks.items()]
pd.DataFrame(rows).to_csv(OUT/'01_auditoria_estatica_original.csv',index=False)
assert all(checks.values()),f'Falha na auditoria estática: {[a for a,v in checks.items() if not v]}'
plan=k[['Dataset','Method','TopK']].copy()
plan['Model']='DecisionTree'
plan['Train_Only_Selection']=True
plan['Preprocessor_Fit_On_Outer_Train']=True
plan['External_Test_Evaluations_Planned']=1
plan['Status']='PLANNED_NOT_EXECUTED'
plan.to_csv(OUT/'02_plano_120_execucoes_externas.csv',index=False)
report={
 'phase':'5C','status':'STATIC_REVIEW_PASSED_NOT_EXTERNAL_EXECUTION_APPROVED',
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'original_benchmark_sha256':source_hash,
 'protocol_sha256':proto['protocol_sha256'],
 'datasets':12,'methods':10,'frozen_k':120,'checks':checks,
 'limitations':[
  'A revisão estática do código interno NÃO comprova o comportamento do código externo ainda não implementado.',
  'É obrigatório ajustar preprocessador, filtro e seletor apenas no outer train.',
  'Os índices externos devem ser carregados dos arquivos já existentes, sem recriar split.',
  'Antes de liberar o teste, auditar o script final de execução externa completo.',
  'Não usar o teste para selecionar K, ajustar hiperparâmetros ou escolher a melhor variante.',
  'Executar cada combinação uma vez; registrar resultados e hashes imutáveis.',
  'O código CFIS original exige cfis_function e adaptador previamente validados no notebook.'
 ]}
(OUT/'00_manifesto_revisao_estatica.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'03_requisitos_para_execucao_externa.md').write_text("""# Checklist obrigatório antes de abrir o teste externo

- [ ] Script final revisado integralmente (esta fase só audita o benchmark interno original).
- [ ] Reutilizar exatamente as funções de ranking e árvore do código original.
- [ ] Preservar CFIS original, adaptador numérico validado e configuração de sementes.
- [ ] Carregar train/test indices preexistentes sem gerar novo split.
- [ ] Ajustar pré-processador e VarianceThreshold exclusivamente em outer train.
- [ ] Ajustar os dez seletores exclusivamente em outer train, com K fixado na Fase 4.
- [ ] Aplicar transformações e colunas selecionadas ao teste apenas para predict.
- [ ] Registrar hashes, tempos e resultados sem sobregravar checkpoints internos.
- [ ] Não escolher hiperparâmetros, K ou método a partir dos resultados externos.
""",encoding='utf-8')
print('FASE 5C CONCLUÍDA — REVISÃO ESTÁTICA DO CÓDIGO ORIGINAL',flush=True)
print('Verificações:',sum(checks.values()),'/',len(checks),flush=True)
print('Plano externo:',len(plan),'combinações | 12 datasets | 10 métodos',flush=True)
print('SHA256 do benchmark original:',source_hash,flush=True)
print('Teste externo: NÃO ACESSADO | Modelos: NÃO TREINADOS',flush=True)
print('Saída:',OUT,flush=True)
print('ATENÇÃO: auditoria do código externo ainda é obrigatória antes da execução.',flush=True)
