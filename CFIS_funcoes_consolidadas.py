"""CFIS completo + funções de permutação históricas extraídas do notebook.
As funções auxiliares são preservadas; execute testes de equivalência antes de usar no benchmark.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from types import SimpleNamespace
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.inspection import permutation_importance
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score


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




def fast_binary_permutation_importance_v2(
    estimator, X, y, n_repeats=10, random_state=42
):
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape
    coef = estimator.coef_[0]
    classes = estimator.classes_

    base_scores = estimator.decision_function(X)
    baseline_accuracy = np.mean(estimator.predict(X) == y)

    # Mesma semente interna utilizada pelo scikit-learn.
    rng = check_random_state(random_state)
    internal_seed = rng.randint(np.iinfo(np.int32).max)

    importances = np.empty(
        (n_features, n_repeats), dtype=float
    )

    for j in range(n_features):

        feature_rng = check_random_state(internal_seed)
        shuffle_idx = np.arange(n_samples)

        # A coluna é modificada cumulativamente.
        current_column = X[:, j].copy()
        original_column = X[:, j]

        for r in range(n_repeats):

            feature_rng.shuffle(shuffle_idx)

            # CORREÇÃO: permutar a coluna já modificada.
            current_column = current_column[shuffle_idx].copy()

            updated_scores = (
                base_scores
                + (current_column - original_column) * coef[j]
            )

            predictions = np.where(
                updated_scores > 0,
                classes[1],
                classes[0]
            )

            accuracy = np.mean(predictions == y)

            importances[j, r] = (
                baseline_accuracy - accuracy
            )

    return importances


def fast_regression_permutation_v4(
    estimator, X, y,
    n_repeats=10,
    random_state=42
):
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    coef = np.asarray(estimator.coef_).ravel()

    if coef.shape != (n_features,):
        raise ValueError("Coeficientes incompatíveis.")

    baseline_predictions = estimator.predict(X)
    baseline_r2 = r2_score(y, baseline_predictions)

    internal_seed = check_random_state(
        random_state
    ).randint(np.iinfo(np.int32).max)

    importances = np.empty(
        (n_features, n_repeats),
        dtype=float
    )

    for j in range(n_features):
        rng = check_random_state(internal_seed)
        indices = np.arange(n_samples)

        original_column = X[:, j]
        current_column = original_column.copy()

        for r in range(n_repeats):
            rng.shuffle(indices)

            # Mesma permutação cumulativa do sklearn.
            current_column = current_column[
                indices
            ].copy()

            predictions = (
                baseline_predictions
                + (current_column - original_column)
                * coef[j]
            )

            permuted_r2 = r2_score(y, predictions)

            importances[j, r] = (
                baseline_r2 - permuted_r2
            )

    return importances


def optimized_regression_adapter_v4(
    estimator,
    X,
    y,
    n_repeats=10,
    random_state=42,
    scoring=None,
    n_jobs=None,
    **kwargs
):
    # Impedir uso fora do escopo validado.
    if scoring != "r2":
        raise ValueError("V4 exige scoring='r2'.")

    if kwargs:
        raise ValueError(
            f"Parâmetros não suportados: {list(kwargs)}"
        )

    if n_jobs not in (None, 1):
        raise ValueError("V4 exige n_jobs=1 ou None.")

    values = fast_regression_permutation_v4(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state
    )

    return SimpleNamespace(
        importances=values,
        importances_mean=values.mean(axis=1),
        importances_std=values.std(axis=1)
    )


def consolidated_permutation_adapter(
    estimator, X, y,
    n_repeats=10,
    random_state=42,
    scoring=None,
    n_jobs=None,
    **kwargs
):
    if kwargs or n_jobs not in (None, 1):
        raise ValueError("Parâmetros fora do escopo validado.")

    X_array = np.asarray(X)

    if scoring == "accuracy":
        if not hasattr(estimator, "classes_"):
            raise ValueError("Estimador sem classes_.")

        n_classes = len(estimator.classes_)

        if n_classes == 2:
            values = fast_binary_permutation_importance_v2(
                estimator, X_array, y,
                n_repeats=n_repeats,
                random_state=random_state
            )
        elif n_classes > 2:
            values = fast_multiclass_permutation_importance_v3(
                estimator, X_array, y,
                n_repeats=n_repeats,
                random_state=random_state
            )
        else:
            raise ValueError("Número de classes inválido.")

    elif scoring == "r2":
        values = fast_regression_permutation_v4(
            estimator, X_array, y,
            n_repeats=n_repeats,
            random_state=random_state
        )

    else:
        raise ValueError(f"Scoring não suportado: {scoring}")

    return SimpleNamespace(
        importances=values,
        importances_mean=values.mean(axis=1),
        importances_std=values.std(axis=1)
    )


def execute_cfis(data, variant):
    if variant == "Original":
        adapter = original_permutation
    else:
        adapter = consolidated_permutation_adapter

    try:
        cfis_globals["permutation_importance"] = adapter

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        result = cfis_function(
            data["X_train"],
            data["y_train"],
            task=data["task"],
            model=None,
            n_repeats=N_REPEATS,
            random_state=SEED
        )

        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = time.perf_counter() - wall_start

    finally:
        cfis_globals["permutation_importance"] = (
            original_permutation
        )

    return result, wall_elapsed, cpu_elapsed


def audit_cfis(original, optimized, task):
    same_order = (
        original["Feature"].tolist()
        == optimized["Feature"].tolist()
    )

    a = original.set_index("Feature")
    b = optimized.set_index("Feature")

    if set(a.index) != set(b.index):
        return False, False, False, float("inf")

    b = b.reindex(a.index)

    exact_components = []
    close_components = []
    max_diff = 0.0

    for column in COMPONENTS:
        av = a[column].to_numpy(dtype=float)
        bv = b[column].to_numpy(dtype=float)

        exact_components.append(
            bool(np.array_equal(av, bv))
        )
        close_components.append(
            bool(np.allclose(
                av, bv, atol=ATOL, rtol=0
            ))
        )
        max_diff = max(
            max_diff,
            float(np.max(np.abs(av - bv)))
        )

    exact = same_order and all(exact_components)
    close = same_order and all(close_components)

    accepted = exact if task == "classification" else close

    return accepted, exact, same_order, max_diff


# Referências exigidas pelo executor. Flags de equivalência não são presumidas.
cfis_function = CFIS_feature_selection
cfis_globals = cfis_function.__globals__
original_permutation = cfis_globals['permutation_importance']
exact_cfis = False
equivalent_1e12 = False

