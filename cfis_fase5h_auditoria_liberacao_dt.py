"""Fase 5H: auditoria estática e do ambiente, SEM acessar o teste externo."""
from pathlib import Path
import ast, json, hashlib, inspect
import pandas as pd

ROOT = Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
RUN = ROOT/'04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1'
BASE = RUN/'Results/Auditoria_Cientifica'
OUT = BASE/'Fase5H_Auditoria_Liberacao_DT'
OUT.mkdir(parents=True, exist_ok=True)
PROTO = RUN/'Manifests/protocol.json'
FROZEN = BASE/'Fase4_Consolidacao_DecisionTree/01_k_congelado_120_combinacoes.csv'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

proto=json.loads(PROTO.read_text(encoding='utf-8'))
k=pd.read_csv(FROZEN)
checks={}
checks['decision_tree']=proto.get('model')=='DecisionTree'
checks['seed_42']=proto.get('seed')==42
checks['protocol_hash_matches']=proto.get('protocol_sha256')=='7e4227be5d6572d3d3c538b6e8124c288a4727405ca920677d02fd24ea5bad83'
checks['12_datasets_10_methods']=len(proto['datasets'])==12 and len(proto['methods'])==10
checks['120_frozen_k']=len(k)==120 and not k[['Dataset','Method']].duplicated().any()
checks['frozen_sets_match']=set(zip(k.Dataset,k.Method))=={(d,m) for d in proto['datasets'] for m in proto['methods']}
checks['frozen_status']=k.K_Status.eq('FROZEN_FROM_INNER_VALIDATION__NO_EXTERNAL_TEST').all()
checks['valid_k']=all(int(r.TopK) in proto['topk'][r.Dataset] for r in k.itertuples())
checks['rankings_120_present']=sum(1 for _ in (RUN/'Checkpoints').glob('*/split_*/ranking__CFIS.json'))==120
required=['cfis_function','cfis_globals','original_permutation','consolidated_permutation_adapter']
missing=[n for n in required if n not in globals()]
checks['cfis_loaded']=not missing
if not missing:
    checks['cfis_namespace']=cfis_function.__globals__ is cfis_globals
    checks['cfis_permutation_restored']=cfis_globals.get('permutation_importance') is original_permutation
else:
    checks['cfis_namespace']=False
    checks['cfis_permutation_restored']=False
# Important: historical reproduction was reported by the user; it is NOT proven by this static cell.
# Static inspection of candidate source (provided by user or uploaded to Colab).
candidate_paths=[Path('/content/cfis_fase5d_candidato_externo_dt.py'), Path('cfis_fase5d_candidato_externo_dt.py')]
candidate=next((p for p in candidate_paths if p.is_file()),None)
checks['candidate_source_available']=candidate is not None
if candidate:
    src=candidate.read_text(encoding='utf-8')
    tree=ast.parse(src)
    assignments={t.id: n.value for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    lock=assignments.get('EXTERNAL_TEST_EXECUTION_ALLOWED')
    checks['candidate_still_locked']=isinstance(lock,ast.Constant) and lock.value is False
    checks['candidate_has_external_job']=any(isinstance(n,ast.FunctionDef) and n.name=='candidate_external_job' for n in tree.body)
    checks['candidate_has_baseline']= 'baseline' in src.lower() and 'all_features' in src.lower()
    checks['random_policy_explicit']= 'SEED+split_no' in src and 'compute_ranking(method,train_mat,ytrain,names,task,0)' not in src
else:
    for key in ['candidate_still_locked','candidate_has_external_job','candidate_has_baseline','random_policy_explicit']:
        checks[key]=False
# These are independent gates requiring review and a separate release script.
checks['historical_equivalence_120_recomputed']=False
checks['external_executor_independently_approved']=False
checks['external_test_accessed_by_this_audit']=False  # informational, not an approval gate

report={'phase':'5H','status':'AUDIT_ONLY__NO_EXTERNAL_RELEASE','protocol_sha256':proto['protocol_sha256'],
        'frozen_k_sha256':sha256(FROZEN),'candidate_source':str(candidate) if candidate else None,
        'candidate_sha256':sha256(candidate) if candidate else None,'checks':checks,
        'cfis_missing':missing,'external_test_accessed':False,'models_trained':False,
        'note':'Esta auditoria NÃO libera o teste externo e não demonstra reprodução dos 120 rankings.'}
(OUT/'00_manifesto_5h.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame([{'Check':name,'Result':bool(value)} for name,value in checks.items()]).to_csv(OUT/'01_verificacoes.csv',index=False)
print('FASE 5H — AUDITORIA DO EXECUTOR (SEM TESTE EXTERNO)')
for name,value in checks.items():
    print(f"{'OK' if value else 'PENDENTE'} | {name}")
print('\nCFIS ausente:',missing)
print('Executor candidato:',candidate if candidate else 'não encontrado no Colab')
print('Manifesto:',OUT/'00_manifesto_5h.json')
print('STATUS: NÃO LIBERADO | Teste externo: NÃO ACESSADO')
