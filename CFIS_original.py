# Backup do CFIS recuperado da saída da célula 2 do Untitled49.ipynb
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.inspection import permutation_importance

def CFIS_feature_selection(
    X,
    y,
    task="classification",
    model=None,
    n_repeats=10,
    random_state=42
):
    """
    Replicação do CFIS fornecido.

    Componentes:
      1. Permutation Importance absoluta
      2. Coeficientes absolutos
      3. ANOVA F (classificação) ou
         Pearson absoluto (regressão)

    Combinação:
        CFIS = (P + C + R) / 3

    Correções operacionais:
      - Modelo ajustado antes da permutação
      - Coeficientes multiclasse agregados
        pela média dos valores absolutos

    IMPORTANTE:
      X deve conter apenas dados de treinamento.
    """

    if task not in ("classification", "regression"):
        raise ValueError("Tarefa inválida.")

    # --------------------------------------------------------
    # 1. VALIDAR ENTRADAS
    # --------------------------------------------------------

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_values = X.to_numpy(dtype=float)
    else:
        X_values = np.asarray(X, dtype=float)
        feature_names = [
            f"Feature_{i}"
            for i in range(X_values.shape[1])
        ]

    y_values = np.asarray(y).ravel()

    if X_values.ndim != 2:
        raise ValueError("X deve ser bidimensional.")

    if len(X_values) != len(y_values):
        raise ValueError("X e y possuem tamanhos diferentes.")

    if not np.isfinite(X_values).all():
        raise ValueError("X contém valores não finitos.")

    if pd.isna(y_values).any():
        raise ValueError("y contém valores ausentes.")

    n_features = X_values.shape[1]

    # --------------------------------------------------------
    # 2. PADRONIZAÇÃO — APENAS TREINAMENTO
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    # --------------------------------------------------------
    # 3. MODELO BASE
    # --------------------------------------------------------

    if model is None:
        if task == "classification":
            estimator = LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        else:
            estimator = Lasso(
                alpha=0.01,
                random_state=random_state,
                max_iter=1000
            )
    else:
        estimator = clone(model)

    # Correção: ajustar antes da permutação.
    estimator.fit(X_scaled, y_values)

    # --------------------------------------------------------
    # 4. COMPONENTE P — PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    scoring = (
        "accuracy"
        if task == "classification"
        else "r2"
    )

    perm_result = permutation_importance(
        estimator,
        X_scaled,
        y_values,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1
    )

    permutation_scores = np.abs(
        perm_result.importances_mean
    )

    permutation_scores = np.nan_to_num(
        permutation_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # 5. COMPONENTE C — COEFICIENTES ABSOLUTOS
    # --------------------------------------------------------

    if hasattr(estimator, "coef_"):

        coef = np.asarray(estimator.coef_)

        if coef.ndim == 1:
            coefficient_scores = np.abs(coef)

        elif coef.ndim == 2:
            # Correção operacional para multiclasse:
            # um valor por característica.
            coefficient_scores = np.mean(
                np.abs(coef),
                axis=0
            )

        else:
            raise ValueError(
                "Formato inesperado de coeficientes."
            )

    elif hasattr(estimator, "feature_importances_"):

        coefficient_scores = np.abs(
            np.asarray(
                estimator.feature_importances_
            )
        )

    else:
        coefficient_scores = np.zeros(n_features)

    if len(coefficient_scores) != n_features:
        raise ValueError(
            "Dimensão dos coeficientes incompatível."
        )

    # --------------------------------------------------------
    # 6. COMPONENTE R — ANOVA OU PEARSON
    # --------------------------------------------------------

    correlation_scores = np.zeros(n_features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if task == "classification":

            classes = np.unique(y_values)

            for j in range(n_features):

                groups = [
                    X_scaled[y_values == cls, j]
                    for cls in classes
                ]

                if (
                    all(len(g) > 0 for g in groups)
                    and np.var(X_scaled[:, j]) > 0
                ):
                    f_value, _ = stats.f_oneway(*groups)

                    if np.isfinite(f_value):
                        correlation_scores[j] = max(
                            0.0,
                            float(f_value)
                        )

        else:

            for j in range(n_features):

                if np.var(X_scaled[:, j]) > 0:

                    corr = np.corrcoef(
                        X_scaled[:, j],
                        y_values
                    )[0, 1]

                    if np.isfinite(corr):
                        correlation_scores[j] = abs(
                            float(corr)
                        )

    # --------------------------------------------------------
    # 7. NORMALIZAÇÃO ORIGINAL POR SOMA
    # --------------------------------------------------------

    def normalize_component(values):

        values = np.nan_to_num(
            np.asarray(values, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        total = np.sum(values)

        if total > 0:
            return values / total

        return np.zeros_like(values)

    normalized_perm = normalize_component(
        permutation_scores
    )

    normalized_coef = normalize_component(
        coefficient_scores
    )

    normalized_corr = normalize_component(
        correlation_scores
    )

    # --------------------------------------------------------
    # 8. COMBINAÇÃO MATEMÁTICA ORIGINAL
    # --------------------------------------------------------

    combined_score = (
        normalized_perm
        + normalized_coef
        + normalized_corr
    ) / 3.0

    cfis_percentage = normalize_component(
        combined_score
    ) * 100.0

    # --------------------------------------------------------
    # 9. RESULTADOS
    # --------------------------------------------------------

    results = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance": normalized_perm,
        "Coefficient_Abs": normalized_coef,
        "Correlation_Abs": normalized_corr,
        "CFIS_Score": combined_score,
        "CFIS_Percentage": cfis_percentage
    })

    return results.sort_values(
        "CFIS_Score",
        ascending=False,
        kind="stable"
    ).reset_index(drop=True)


