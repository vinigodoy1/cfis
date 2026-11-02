# CFIS — Fase 5B: auditoria estrutural da avaliação externa Decision Tree.
# NÃO lê amostras/labels do teste, NÃO treina, NÃO executa seletores.
from pathlib import Path
import json, hashlib, datetime, traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
import pyarrow.parquet as pq

ROOT=Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
DATA=ROOT/'01_Datasets'
RUN=ROOT/'04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1'
BASE=RUN/'Results/Auditoria_Cientifica'
OUT=BASE/'Fase5B_Auditoria_Pipeline_DT'
OUT.mkdir(parents=True,exist_ok=True)
PROTO=RUN/'Manifests/protocol.json'
KFILE=BASE/'Fase4_Consolidacao_DecisionTree/01_k_congelado_120_combinacoes.csv'
PREF=BASE/'Fase5_Preflight_Externo_DT/01_preflight_12_datasets.csv'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check(cond,msg):
    if not cond:raise ValueError(msg)

for p in (PROTO,KFILE,PREF):check(p.is_file(),f'Arquivo ausente: {p}')
protocol=json.loads(PROTO.read_text(encoding='utf-8'))
k=pd.read_csv(KFILE)
pref=pd.read_csv(PREF)
check(protocol.get('model')=='DecisionTree' and protocol.get('seed')==42,'Protocolo incompatível')
check(len(k)==120 and len(pref)==12 and pref.Status.eq('OK').all(),'Fase 5A ou K congelado inválidos')
check(not k[['Dataset','Method']].duplicated().any(),'Pares de K duplicados')
check(set(k.Dataset)==set(protocol['datasets']) and set(k.Method)==set(protocol['methods']),'Métodos/datasets divergentes')
check(k.K_Status.eq('FROZEN_FROM_INNER_VALIDATION__NO_EXTERNAL_TEST').all(),'K não congelado')

rows=[]
print('FASE 5B — AUDITORIA ESTRUTURAL DO PIPELINE EXTERNO (SEM ACESSAR AMOSTRAS DO TESTE)',flush=True)
for i,ds in enumerate(protocol['datasets'],1):
    issues=[]; details={}
    try:
        source=next((DATA/d/f'{ds}.parquet' for d in ('Raw','Synthetic') if (DATA/d/f'{ds}.parquet').is_file()),None)
        check(source is not None,f'Parquet não localizado: {ds}')
        meta=json.loads(source.with_name(f'{ds}_metadata.json').read_text(encoding='utf-8'))
        # Somente o rodapé/metadata do Parquet; nenhuma linha ou rótulo é carregado.
        pf=pq.ParquetFile(source)
        nrows=int(pf.metadata.num_rows)
        columns=pf.schema_arrow.names
        target=meta['target_column']
        check(target in columns,f'Coluna alvo ausente no schema: {target}')
        check(len(columns)>1,'Parquet sem características')
        tr=np.load(DATA/'Splits'/f'{ds}_train_indices.npy',allow_pickle=False)
        te=np.load(DATA/'Splits'/f'{ds}_test_indices.npy',allow_pickle=False)
        for label,a in [('train',tr),('test',te)]:
            check(a.ndim==1 and np.issubdtype(a.dtype,np.integer),f'Índices {label} inválidos')
            check(len(a)>0 and len(np.unique(a))==len(a),f'Índices {label} vazios/duplicados')
            check(np.min(a)>=0 and np.max(a)<nrows,f'Índices {label} fora do Parquet')
        check(np.intersect1d(tr,te).size==0,'SOBREPOSIÇÃO treino/teste')
        check(len(tr)+len(te)==nrows,'Treino e teste não cobrem todas as linhas')
        check(np.array_equal(np.sort(np.r_[tr,te]),np.arange(nrows)),'Partição não cobre exatamente o Parquet')
        check(len(tr)==int(pref.loc[pref.Dataset.eq(ds),'Train_Rows'].iloc[0]),'Contagem de treino difere da 5A')
        check(len(te)==int(pref.loc[pref.Dataset.eq(ds),'Test_Rows'].iloc[0]),'Contagem de teste difere da 5A')
        if ds=='20_Newsgroups':check('text' in columns,'Coluna text ausente')
        # Somente arquivos joblib de origem confiável do próprio projeto.
        artifact=joblib.load(DATA/'Processed'/ds/'preprocessor.joblib')
        check(isinstance(artifact,dict),'Artefato de pré-processamento não é dict')
        check('preprocessor' in artifact and 'variance_filter' in artifact,'Componentes do pré-processamento ausentes')
        prep=clone(artifact['preprocessor']); vf=clone(artifact['variance_filter'])
        check(hasattr(prep,'fit_transform') and hasattr(prep,'transform'),'Pré-processador incompatível')
        check(hasattr(vf,'fit_transform') and hasattr(vf,'transform'),'Filtro de variância incompatível')
        # A clonagem garante objetos novos para o fit exclusivo no treino; nenhum fit nesta auditoria.
        check(prep is not artifact['preprocessor'] and vf is not artifact['variance_filter'],'Clone não independente')
        kd=k[k.Dataset.eq(ds)]
        check(len(kd)==10 and set(kd.Method)==set(protocol['methods']),'K por dataset incompleto')
        check(all(int(v) in set(map(int,protocol['topk'][ds])) for v in kd.TopK),'K fora da grade congelada')
        details=dict(Parquet_Rows=nrows,Train_Rows=len(tr),Test_Rows=len(te),Raw_Features=len(columns)-1,
                     Target_Column=target,Preprocessor_Class=type(prep).__name__,Variance_Filter_Class=type(vf).__name__,
                     K_Min=int(kd.TopK.min()),K_Max=int(kd.TopK.max()),Methods=len(kd))
    except Exception as e:
        issues.append(f'{type(e).__name__}: {e}')
    status='OK' if not issues else 'REVISAR'
    rows.append({'Dataset':ds,'Status':status,**details,'Issues':' | '.join(issues)})
    print(f'[{i:02d}/12] {ds}: {status}'+(f' — {issues[0]}' if issues else f' | linhas={details["Parquet_Rows"]} | K={details["K_Min"]}..{details["K_Max"]}'),flush=True)

result=pd.DataFrame(rows)
result.to_csv(OUT/'01_auditoria_estrutural_12_datasets.csv',index=False)
passed=bool(result.Status.eq('OK').all())
report={
 'phase':'5B — auditoria estrutural externa Decision Tree',
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'model':'DecisionTree','datasets_ok':int(result.Status.eq('OK').sum()),
 'datasets_total':len(result),'frozen_k':len(k),'structural_checks_passed':passed,
 'protocol_sha256':protocol['protocol_sha256'],
 'input_file_sha256':{'protocol':sha256(PROTO),'k_frozen':sha256(KFILE),'preflight':sha256(PREF)},
 'external_test_samples_read':False,'models_fitted':False,'selectors_fitted':False,
 'limitations':[
  'O rodapé do Parquet foi consultado para obter o schema e o número de linhas; nenhuma amostra do teste foi carregada.',
  'A existência de índices válidos não demonstra, isoladamente, que não houve vazamento na criação dos arquivos.',
  'O código de seleção/treinamento externo ainda deve ser auditado antes de liberar o teste.',
  'A função CFIS original e seu adaptador numérico devem ser reutilizados e validados sem alterações.',
  'O teste externo deve ser avaliado uma única vez após congelar toda a implementação.'
 ]
}
(OUT/'02_relatorio_5b.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('\n'+'='*75)
print('FASE 5B CONCLUÍDA — AUDITORIA ESTRUTURAL DO PIPELINE DECISION TREE')
print(f'Datasets aprovados: {report["datasets_ok"]}/12 | K congelados: {len(k)}')
print('Amostras do teste externo: NÃO LIDAS | Modelos: NÃO TREINADOS')
print('Saída:',OUT)
if not passed:raise RuntimeError('Fase 5B NÃO APROVADA; consultar 01_auditoria_estrutural_12_datasets.csv')
print('Próximo passo: revisão estática e preparação do código de avaliação externa, ANTES de liberar teste.')
