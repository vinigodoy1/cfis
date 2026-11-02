# CFIS — Fase 5A: preflight do teste externo (SEM abrir amostras do teste).
# Executar no Google Colab com Drive montado. Somente leitura dos insumos.
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import pandas as pd

ROOT=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
DATA=ROOT/'01_Datasets'
RUN=ROOT/'04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1'
PROTO=RUN/'Manifests/protocol.json'
MANIFEST=RUN/'Manifests/inner_benchmark_manifest.csv'
FROZEN=RUN/'Results/Auditoria_Cientifica/Fase4_Consolidacao_DecisionTree/01_k_congelado_120_combinacoes.csv'
OUT=RUN/'Results/Auditoria_Cientifica/Fase5_Preflight_Externo_DT'
OUT.mkdir(parents=True,exist_ok=True)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def require(cond,msg):
    if not cond: raise RuntimeError(msg)

print('FASE 5A — PRÉ-VALIDAÇÃO EXTERNA (TESTE BLOQUEADO)',flush=True)
for p in (PROTO,MANIFEST,FROZEN):
    require(p.is_file(),f'Insumo obrigatório ausente: {p}')
proto=json.loads(PROTO.read_text(encoding='utf-8'))
manifest=pd.read_csv(MANIFEST)
k=pd.read_csv(FROZEN)
require(proto.get('model')=='DecisionTree','Modelo diferente de DecisionTree')
require(proto.get('seed')==42,'Seed diferente de 42')
require(len(manifest)==8000,'Manifesto interno não tem 8000 linhas')
require(manifest.Protocol_SHA256.nunique()==1 and manifest.Protocol_SHA256.iloc[0]==proto['protocol_sha256'],'Hash do protocolo divergente')
require(len(k)==120 and not k[['Dataset','Method']].duplicated().any(),'K congelado não tem 120 pares únicos')
require(set(k.Dataset)==set(proto['datasets']),'Datasets congelados divergentes')
require(set(k.Method)==set(proto['methods']),'Métodos congelados divergentes')
require(k.TopK.notna().all() and (k.TopK.astype(int)>0).all(),'K inválido')
require('K_Status' in k and k.K_Status.eq('FROZEN_FROM_INNER_VALIDATION__NO_EXTERNAL_TEST').all(),'K não identificado como congelado')

rows=[]
for i,ds in enumerate(proto['datasets'],1):
    errors=[]
    source=next((DATA/folder/f'{ds}.parquet' for folder in ('Raw','Synthetic') if (DATA/folder/f'{ds}.parquet').is_file()),None)
    metadata=source.with_name(f'{ds}_metadata.json') if source else None
    train_idx=DATA/'Splits'/f'{ds}_train_indices.npy'
    test_idx=DATA/'Splits'/f'{ds}_test_indices.npy'
    prep=DATA/'Processed'/ds/'preprocessor.joblib'
    for label,path in [('source',source),('metadata',metadata),('train_indices',train_idx),('test_indices',test_idx),('preprocessor',prep)]:
        if path is None or not path.is_file():errors.append(f'{label} ausente')
    ntrain=ntest=None
    if train_idx.is_file() and test_idx.is_file():
        tr=np.load(train_idx,allow_pickle=False)
        te=np.load(test_idx,allow_pickle=False)
        for name,arr in [('train',tr),('test',te)]:
            if arr.ndim!=1 or not np.issubdtype(arr.dtype,np.integer):errors.append(f'índices {name} inválidos')
            elif len(np.unique(arr))!=len(arr):errors.append(f'índices {name} duplicados')
            elif (arr<0).any():errors.append(f'índices {name} negativos')
        if tr.ndim==1 and te.ndim==1 and np.issubdtype(tr.dtype,np.integer) and np.issubdtype(te.dtype,np.integer):
            if np.intersect1d(tr,te).size:errors.append('VAZAMENTO: índices treino/teste sobrepostos')
        ntrain,ntest=len(tr),len(te)
    # Não lemos o parquet nem qualquer amostra do teste nesta fase.
    if metadata and metadata.is_file():
        meta=json.loads(metadata.read_text(encoding='utf-8'))
        if not meta.get('target_column'):errors.append('target_column ausente no metadata')
    kd=k.loc[k.Dataset.eq(ds)]
    if len(kd)!=10:errors.append('esperados 10 K congelados')
    allowed=set(map(int,proto['topk'][ds]))
    if not set(kd.TopK.astype(int)).issubset(allowed):errors.append('K congelado fora da grade original')
    status='OK' if not errors else 'REVISAR'
    rows.append(dict(Dataset=ds,Status=status,Train_Rows=ntrain,Test_Rows=ntest,Methods=len(kd),K_Min=int(kd.TopK.min()),K_Max=int(kd.TopK.max()),Issues='; '.join(errors),Source=str(source) if source else '',Preprocessor=str(prep)))
    print(f'[{i:02d}/12] {ds}: {status} | treino={ntrain} | teste={ntest}'+(f' | {errors}' if errors else ''),flush=True)

report=pd.DataFrame(rows)
report.to_csv(OUT/'01_preflight_12_datasets.csv',index=False)
k[['Dataset','Method','TopK']].sort_values(['Dataset','Method']).to_csv(OUT/'02_k_bloqueados_120.csv',index=False)
summary={
    'stage':'Fase 5A — preflight; no external test sample read',
    'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'model':'DecisionTree','datasets':12,'methods':10,'frozen_k':120,
    'protocol_sha256':proto['protocol_sha256'],
    'inputs_sha256':{'protocol':digest(PROTO),'inner_manifest':digest(MANIFEST),'frozen_k':digest(FROZEN)},
    'ready_for_pipeline_code_audit':bool(report.Status.eq('OK').all()),
    'limitations':['Parquet content and test labels deliberately not read','Preprocessor joblib not deserialized','No final feature ranking or model fit executed','Index-to-parquet row bounds and target schema must be verified in subsequent controlled audit','Final external evaluation is NOT authorized by this report alone']
}
(OUT/'03_relatorio_preflight.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n'+'='*72)
print('FASE 5A CONCLUÍDA — PRÉ-VALIDAÇÃO EXTERNA DECISION TREE')
print(f'Datasets OK: {int(report.Status.eq("OK").sum())}/12 | K congelados: {len(k)}')
print('Teste externo: NÃO ACESSADO | Modelos: NÃO TREINADOS')
print('Resultados:',OUT)
if not report.Status.eq('OK').all():
    raise RuntimeError('PRÉ-VALIDAÇÃO NÃO APROVADA: verifique 01_preflight_12_datasets.csv')
print('Próximo passo: auditar a execução final, inclusive índices versus parquet e fit exclusivo no treino.')
