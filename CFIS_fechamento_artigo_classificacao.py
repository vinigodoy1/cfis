# CFIS — Fechamento científico para o artigo Decision Tree Classification
# Executar no Google Colab após montar o Drive. Não treina, não seleciona K, não reavalia teste externo.
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN = Path("/content/drive/MyDrive/CFIS_Artigo_Diogo/04_Efficiency/CFIS_Final_Benchmark/03_Final_Execution/decision_tree_final_v1")
AUD = RUN / "Results/Auditoria_Cientifica"
FROZEN = AUD / "Fase4_Consolidacao_DecisionTree"
BENCH = RUN / "Results/Benchmark_Final_Classification_10"
OUT = BENCH / "Fechamento_Artigo"
OUT.mkdir(parents=True, exist_ok=True)
REG = {"California_Housing", "Diabetes_Reg"}

def load(path, required):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: colunas ausentes {sorted(missing)}. Encontradas: {list(df.columns)}")
    return df

def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"[OK] {name}: {len(df)} linhas")
    return df

def classification(df):
    return df.loc[~df.Dataset.isin(REG)].copy()

def mean_valid(x):
    return x.mean()

st = classification(load(AUD/"06_estabilidade.csv", ["Dataset","Method","TopK","Mean_Jaccard","Nogueira","Feature_Universe_Consistent"]))
ks = classification(load(FROZEN/"01_k_congelado_120_combinacoes.csv", ["Dataset","Method","TopK"]))
con = classification(load(AUD/"07_concordancia_cfis.csv", ["Dataset","Split","Competitor","TopK","Jaccard","Precision_at_K","Spearman","Kendall","Same_Feature_Universe"]))
eff = classification(load(AUD/"02_eficiencia_selecao.csv", ["Dataset","Split","Method","Selection_Wall_Seconds","Selection_CPU_Seconds","Selection_Peak_RSS_MiB","Energy_Wh"]))
ext = classification(load(BENCH/"Tabelas/01_Resultados_Externos_Classificacao.csv", ["Dataset","Method","TopK","Macro_F1","Selection_s","Total_s"]))
rank = load(BENCH/"Tabelas/03_Ranking_Final.csv", ["Method","Mean_Rank","Mean_F1"])
stats = load(BENCH/"Tabelas/07_Wilcoxon_Holm.csv", ["Competitor","Wilcoxon_p","Holm_p"])
assert st.Dataset.nunique() == 10 and ks.Dataset.nunique() == 10
assert len(ks)==100 and len(eff)==1000 and len(ext)==110, (len(ks),len(eff),len(ext))
assert not st.duplicated(["Dataset","Method","TopK"]).any()
assert not ks.duplicated(["Dataset","Method"]).any()

# A estabilidade foi calculada a partir dos rankings dos 10 splits internos.
# Este script reutiliza os resultados históricos e não afirma ter recalculado os conjuntos.
st["Nogueira_Status"] = np.where(st.Nogueira.notna(), "VALIDO",
    np.where(st.Feature_Universe_Consistent.astype(str).str.lower().isin(["true","1"]),
             "INDEFINIDO_OU_DEGENERADO","UNIVERSO_INCONSISTENTE"))
st["Nogueira_Valido"] = st.Nogueira.where(st.Nogueira_Status.eq("VALIDO"))
save(st, "01_Estabilidade_Todos_K_10_Datasets.csv")

# Análise no K congelado: descritiva, K difere entre métodos
chosen = ks.merge(st, on=["Dataset","Method","TopK"], validate="one_to_one")
assert len(chosen)==100
save(chosen, "02_Estabilidade_K_Congelado_100.csv")
stab_summary = chosen.groupby("Method",as_index=False).agg(
    N_Datasets=("Dataset","nunique"),
    Mean_Jaccard=("Mean_Jaccard","mean"),
    SD_Jaccard=("Mean_Jaccard","std"),
    Mean_Nogueira_Valid=("Nogueira_Valido","mean"),
    N_Nogueira_Valid=("Nogueira_Valido","count"),
    Mean_Selected_K=("TopK","mean")
).sort_values("Mean_Jaccard",ascending=False)
save(stab_summary, "03_Estabilidade_Resumo_K_Congelado.csv")
save(chosen.groupby(["Method","Nogueira_Status"],as_index=False).size().rename(columns={"size":"N"}),
     "04_Diagnostico_Nogueira.csv")

# Comparação em K comum: por dataset, usa somente K presente para TODOS os métodos.
# Não inventa novos rankings ou resultados.
common_rows=[]
coverage=[]
for ds,g in st.groupby("Dataset"):
    methods=set(ks.loc[ks.Dataset.eq(ds),"Method"])
    k_sets=[set(g.loc[g.Method.eq(m),"TopK"].dropna().astype(int)) for m in methods]
    common=sorted(set.intersection(*k_sets)) if k_sets else []
    coverage.append({"Dataset":ds,"Methods":len(methods),"Common_K":",".join(map(str,common)),
                     "N_Common_K":len(common)})
    for k in common:
        common_rows.append(g.loc[g.TopK.eq(k)].copy())
coverage=save(pd.DataFrame(coverage),"05_Cobertura_K_Comum.csv")
if common_rows:
    st_common=pd.concat(common_rows,ignore_index=True)
    save(st_common,"06_Estabilidade_K_Comum_Detalhe.csv")
    common_summary=st_common.groupby(["Dataset","Method"],as_index=False).agg(
        Mean_Jaccard_Across_Common_K=("Mean_Jaccard","mean"),
        Mean_Nogueira_Valid_Across_Common_K=("Nogueira_Valido","mean"),
        N_Common_K=("TopK","nunique"))
    save(common_summary,"07_Estabilidade_K_Comum_Resumo.csv")
else:
    common_summary=pd.DataFrame()
    print("[ATENÇÃO] Nenhum K comum aos 10 métodos; não produzir ranking comparativo de estabilidade em K igual.")

# Concordância CFIS x estimadores: separar por comparador, sem confundir concordância com verdade.
save(con,"08_Concordancia_Detalhe.csv")
con_summary=con.groupby(["Competitor"],as_index=False).agg(
    N=("Jaccard","size"),
    Mean_Jaccard=("Jaccard","mean"),
    Mean_Precision_at_K=("Precision_at_K","mean"),
    Mean_Spearman=("Spearman","mean"),
    Mean_Kendall=("Kendall","mean"),
    N_Universo_Consistente=("Same_Feature_Universe",lambda x: x.astype(str).str.lower().isin(["true","1"]).sum())
)
save(con_summary,"09_Concordancia_Resumo.csv")

# Eficiência dos rankings nos splits internos, separada da execução externa.
save(eff,"10_Eficiencia_Interna_10_Datasets.csv")
eff_summary=eff.groupby(["Dataset","Method"],as_index=False).agg(
    Mean_Wall_s=("Selection_Wall_Seconds","mean"),
    SD_Wall_s=("Selection_Wall_Seconds","std"),
    Mean_CPU_s=("Selection_CPU_Seconds","mean"),
    Mean_Peak_RSS_MiB=("Selection_Peak_RSS_MiB","mean"),
    Mean_Energy_Wh=("Energy_Wh","mean"),
    N_Splits=("Split","nunique"))
save(eff_summary,"11_Eficiencia_Interna_Por_Dataset_Metodo.csv")
eff_methods=eff_summary.groupby("Method",as_index=False).agg(
    Mean_Wall_s=("Mean_Wall_s","mean"),
    Mean_CPU_s=("Mean_CPU_s","mean"),
    Mean_Peak_RSS_MiB=("Mean_Peak_RSS_MiB","mean"),
    Mean_Energy_Wh=("Mean_Energy_Wh","mean"),
    N_Datasets=("Dataset","nunique"))
save(eff_methods,"12_Eficiencia_Interna_Resumo.csv")

# Eficiência externa e resultado externo no K escolhido por validação interna.
save(ext,"13_Desempenho_Eficiencia_Externa.csv")
joined=rank.merge(stab_summary,on="Method",validate="one_to_one").merge(eff_methods,on="Method",validate="one_to_one")
save(joined,"14_Tabela_Integrada_Artigo.csv")

# Figuras com resultados medidos (sem escala truncada)
fig,ax=plt.subplots(figsize=(10,5))
r=rank.sort_values("Mean_Rank",ascending=False)
ax.barh(r.Method,r.Mean_Rank)
ax.set_xlabel("Posto médio externo (menor é melhor)")
ax.set_title("Decision Tree — 10 datasets de classificação")
fig.tight_layout()
fig.savefig(OUT/"Figura_01_Ranking_Externo.png",dpi=200)
plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5))
e=eff_methods.sort_values("Mean_Wall_s",ascending=False)
ax.barh(e.Method,e.Mean_Wall_s)
ax.set_xlabel("Tempo médio de seleção interna (s)")
ax.set_title("Custo de seleção nos splits internos")
fig.tight_layout()
fig.savefig(OUT/"Figura_02_Eficiencia_Interna.png",dpi=200)
plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5))
s=stab_summary.sort_values("Mean_Jaccard",ascending=False)
ax.barh(s.Method,s.Mean_Jaccard)
ax.set_xlabel("Jaccard médio entre splits (K congelado, variável entre métodos)")
ax.set_title("Estabilidade descritiva — K congelado")
fig.tight_layout()
fig.savefig(OUT/"Figura_03_Estabilidade_K_Congelado.png",dpi=200)
plt.close(fig)

# Pacote Excel para revisão de artigo
with pd.ExcelWriter(OUT/"CFIS_Tabelas_Finais_Artigo.xlsx",engine="openpyxl") as writer:
    for sheet,frame in [
        ("Ranking_Externo",rank),("Wilcoxon_Holm",stats),
        ("Estabilidade_K_Cong",chosen),("Estabilidade_Resumo",stab_summary),
        ("Nogueira_Diagnostico",chosen[["Dataset","Method","TopK","Nogueira","Nogueira_Status"]]),
        ("Cobertura_K_Comum",coverage),("Concordancia",con_summary),
        ("Eficiencia_Interna",eff_methods),("Tabela_Integrada",joined)]:
        frame.to_excel(writer,sheet_name=sheet,index=False)

# Relatório científico fiel aos resultados, sem alegar superioridade significativa.
cf=rank.loc[rank.Method.eq("CFIS")].iloc[0]
fpath=BENCH/"Tabelas/06_Friedman.csv"
fr=load(fpath,["Friedman_Chi2","Friedman_p","Kendall_W"]).iloc[0]
report=f"""# CFIS — fechamento científico para redação

## Escopo e proveniência
- Classificação com Decision Tree; 10 datasets; 10 métodos de seleção.
- 110 avaliações externas: 100 seletores e 10 baselines com todas as características.
- K definido anteriormente pelos 10 splits internos; nenhuma seleção posterior pelo teste externo.
- Este recorte para classificação foi decidido após a análise do benchmark de 12 datasets; declarar explicitamente no artigo.
- Resultados de regressão permanecem preservados no benchmark original.

## Resultados preditivos
- CFIS: posto médio {cf.Mean_Rank:.4f}; Macro-F1 médio {cf.Mean_F1:.6f}.
- Friedman: chi²={fr.Friedman_Chi2:.6f}, p={fr.Friedman_p:.8f}, W={fr.Kendall_W:.6f}.
- Não há evidência de diferença global estatisticamente significativa (alpha=0,05).
- Comparações pareadas Holm: consultar 07_Wilcoxon_Holm.csv; não afirmar superioridade sem evidência.

## Estabilidade
- Fonte: rankings dos 10 splits internos, para cada dataset, método e K.
- A estabilidade no K congelado é descritiva: K pode diferir entre métodos.
- Datasets com K comum a todos os métodos: {int((coverage.N_Common_K>0).sum())}/10.
- Índice de Nogueira: ausências mantidas como ausências; universos inconsistentes não são alinhados artificialmente.
- Interpretar Jaccard com cuidado se o universo de características variar entre splits.

## Concordância
- Comparações CFIS versus métodos concorrentes, sem tratar os métodos de referência como verdade-terreno.
- As métricas são concordância entre rankings/conjuntos, não avaliação externa de acurácia.

## Eficiência
- Interna: tempo de parede, CPU, pico RSS e energia estimada dos 10 splits.
- Externa: tempo de seleção, treino, predição e total no conjunto de teste externo.
- Não misturar tempo interno repetido com tempo externo de uma única avaliação.
- RSS é memória do processo amostrada, não necessariamente pico de toda a máquina; energia é estimativa.

## Limitações obrigatórias
- Um split externo por dataset: não estimar variância de teste por dataset a partir dos 10 splits internos.
- Apenas Decision Tree: não extrapolar para Random Forest ou outros modelos.
- Classificação/regressão não misturadas na inferência principal.
- Escolha pós-hoc do escopo de classificação; resultados de regressão arquivados.
- Comparação em K variável não isola estabilidade intrínseca dos métodos.
- Friedman sem significância: resultados de Wilcoxon/Holm devem ser interpretados com cautela.
- Nenhuma ablação dos componentes do CFIS foi realizada neste fechamento.

## Próxima etapa
Redigir Introdução, Trabalhos Relacionados, Método, Protocolo Experimental,
Resultados, Discussão, Ameaças à Validade e Conclusão usando os CSVs desta pasta.
"""
(OUT/"LEIA_ME_Para_Artigo.md").write_text(report,encoding="utf-8")
manifest={"inputs":{},"outputs":sorted(x.name for x in OUT.iterdir()),
          "classification_datasets":sorted(ext.Dataset.unique().tolist()),
          "n_external":len(ext),"n_internal_efficiency":len(eff),
          "n_stability":len(st),"n_concordance":len(con)}
for f in [AUD/"06_estabilidade.csv",AUD/"07_concordancia_cfis.csv",
          AUD/"02_eficiencia_selecao.csv",FROZEN/"01_k_congelado_120_combinacoes.csv",
          BENCH/"Tabelas/01_Resultados_Externos_Classificacao.csv"]:
    manifest["inputs"][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(OUT/"manifesto_fontes.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

print("\n" + "="*70)
print("FECHAMENTO PARA O ARTIGO CONCLUÍDO — SEM NOVOS EXPERIMENTOS")
print("Estabilidade por K:",len(st),"| Estabilidade no K congelado:",len(chosen))
print("Nogueira válido no K congelado:",int(chosen.Nogueira.notna().sum()),"/100")
print("Concordâncias:",len(con),"| Eficiência interna:",len(eff))
print("Datasets com K comum:",int((coverage.N_Common_K>0).sum()),"/10")
print("Saída:",OUT)
print("="*70)
