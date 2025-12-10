"""
lca_radar.py
RAP (Reproducible Analytical Pipeline) para LCA (Latent Class Analysis) em dados binários,
com gráficos radar por classe. Focado em interpretabilidade para profissionais de saúde.

- Leitura direta de Parquet
- Autodetecção de colunas binárias (0/1/NaN/True/False)
- LCA com StepMix (medição Bernoulli), nº de classes parametrizável (default=13)
- Perfis de classe calculados via posterior probabilities (robusto)
- Exporta PNGs dos radares e CSVs com perfis e pesos de classe

Requisitos:
    pip install stepmix pandas numpy matplotlib scikit-learn pyarrow fastparquet
"""

from __future__ import annotations
import os
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stepmix.stepmix import StepMix

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Utilidades de dados
# -----------------------------
def read_parquet_binary(
    parquet_path: str,
    feature_cols: Optional[Sequence[str]] = None,
    id_cols: Optional[Sequence[str]] = None,
    drop_na_rows: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lê um Parquet e retorna (df, cols_binarias) prontos para LCA.

    - Se feature_cols for None: autodetecta colunas binárias (0/1, True/False; aceita NaN).
    - Converte bool -> 0/1.
    - id_cols (se fornecido) são preservadas, mas excluídas da seleção automática.

    Args:
        parquet_path: caminho do arquivo .parquet
        feature_cols: lista explícita de colunas a usar como binárias
        id_cols: colunas de identificação a preservar e não usar na LCA
        drop_na_rows: se True, dropa linhas com NaN nas features; senão, preenche com 0

    Returns:
        df: DataFrame completo (inclui id_cols se existirem)
        binary_cols: lista final de colunas binárias usadas na LCA
    """
    df = pd.read_parquet(parquet_path)

    if id_cols:
        id_cols = [c for c in id_cols if c in df.columns]
    else:
        id_cols = []

    if feature_cols is not None:
        # Usa exatamente o que foi passado
        binary_cols = [c for c in feature_cols if c in df.columns]
        _coerce_bools_inplace(df, binary_cols)
    else:
        # Autodetecta binárias
        candidate_cols = [c for c in df.columns if c not in id_cols]
        binary_cols = _autodetect_binary_columns(df, candidate_cols)
        _coerce_bools_inplace(df, binary_cols)

    # Lidar com NaNs nas features binárias
    if drop_na_rows:
        df = df.dropna(subset=binary_cols)
    else:
        df[binary_cols] = df[binary_cols].fillna(0)

    # Garantir tipo numérico 0/1
    df[binary_cols] = df[binary_cols].astype(float)

    return df, binary_cols


def _coerce_bools_inplace(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Converte colunas booleanas para 0/1 inplace."""
    for c in cols:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)


def _is_binary_series(s: pd.Series) -> bool:
    """True se a série é binária (aceita {0,1}, {True,False}, com ou sem NaN)."""
    vals = pd.Series(s.dropna().unique())
    if vals.empty:
        return False
    # Normaliza bool -> int
    if pd.api.types.is_bool_dtype(vals):
        return True
    try:
        numeric_vals = pd.to_numeric(vals, errors="coerce")
    except Exception:
        return False
    unique_set = set(numeric_vals.dropna().astype(int).unique().tolist())
    return unique_set.issubset({0, 1}) and len(unique_set) > 0


def _autodetect_binary_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """Retorna colunas binárias detectadas automaticamente."""
    binary = [c for c in cols if _is_binary_series(df[c])]
    if len(binary) == 0:
        raise ValueError(
            "Nenhuma coluna binária detectada. "
            "Informe 'feature_cols' manualmente ou revise o dataset."
        )
    return binary


# -----------------------------
# LCA com StepMix
# -----------------------------
def fit_lca_bernoulli(
    X: Union[pd.DataFrame, np.ndarray],
    n_classes: int = 13,
    n_init: int = 5,
    max_iter: int = 500,
    random_state: int = 42,
    verbose: int = 0,
) -> StepMix:
    """
    Ajusta um modelo LCA (Bernoulli) com StepMix.

    Args:
        X: matriz (n_amostras, n_variáveis) com 0/1
        n_classes: número de classes latentes (default 13)
        n_init: quantas inicializações aleatórias
        max_iter: iterações do EM
        random_state: semente
        verbose: 0 (silencioso) a 2 (bem verboso)

    Returns:
        modelo StepMix ajustado
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    model = StepMix(
        n_components=n_classes,
        measurement="bernoulli",   # variáveis observadas binárias
        structural=None,           # sem regressão/estrutura adicional
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(X)
    return model


def class_profiles_from_posteriors(
    X: Union[pd.DataFrame, np.ndarray],
    posteriors: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Calcula, para cada classe, a probabilidade de '1' em cada variável (perfil de classe),
    usando médias ponderadas pelas probabilidades posteriores q(z=c | x).

    Args:
        X: matriz binária (n_amostras, n_variáveis)
        posteriors: matriz (n_amostras, n_classes)
        feature_names: nomes das variáveis (len = n_variáveis)

    Returns:
        DataFrame de shape (n_classes, n_variáveis) com probabilidades em [0,1]
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    n_samples, n_features = X.shape
    n_classes = posteriors.shape[1]
    profiles = np.zeros((n_classes, n_features), dtype=float)

    for c in range(n_classes):
        w = posteriors[:, c].reshape(-1, 1)  # (n,1)
        denom = w.sum()
        # evita divisão por zero
        if denom <= 0:
            profiles[c, :] = np.nan
        else:
            profiles[c, :] = (w * X).sum(axis=0) / denom

    prof_df = pd.DataFrame(profiles, columns=feature_names)
    prof_df.index = [f"Classe_{i+1}" for i in range(n_classes)]
    return prof_df


# -----------------------------
# Radar plots
# -----------------------------
def plot_radar_per_class(
    class_profiles: pd.DataFrame,
    order_features_by: str = "input",  # "input" | "variance" | "mean"
    max_vars: Optional[int] = None,
    output_dir: str = "lca_outputs",
    basename: str = "radar_class",
    save_formats: Tuple[str, ...] = ("png",),
    dpi: int = 150,
) -> List[str]:
    """
    Gera um gráfico radar para cada classe e salva em arquivo.

    Args:
        class_profiles: DF (n_classes x n_features) com probabilidades em [0,1]
        order_features_by: ordenação dos eixos: "input" (sem mudar),
                           "variance" (maior var primeiro) ou "mean" (maior média primeiro)
        max_vars: se quiser limitar o número de variáveis mais informativas
        output_dir: pasta de saída
        basename: prefixo dos arquivos
        save_formats: formatos de imagem, ex. ("png", "pdf")
        dpi: resolução

    Returns:
        Lista com caminhos dos arquivos gerados
    """
    os.makedirs(output_dir, exist_ok=True)

    prof = class_profiles.copy()

    # Ordenação dos eixos (mesmo para todas as classes, para facilitar leitura)
    if order_features_by == "variance":
        order = prof.var(axis=0).sort_values(ascending=False).index.tolist()
    elif order_features_by == "mean":
        order = prof.mean(axis=0).sort_values(ascending=False).index.tolist()
    else:
        order = list(prof.columns)

    if max_vars is not None and max_vars > 0:
        order = order[:max_vars]

    prof = prof[order]

    # Preparar ângulos do radar
    labels = prof.columns.tolist()
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # fecha o ciclo

    out_files: List[str] = []

    for idx, (cls, row) in enumerate(prof.iterrows(), start=1):
        values = row.values.tolist()
        # fecha o ciclo para plotagem
        values += values[:1]

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Eixos e rótulos
        plt.xticks(angles[:-1], labels, fontsize=9)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=8)
        ax.set_ylim(0, 1)

        # Linha e preenchimento
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.2)

        plt.title(f"{cls} — Probabilidade de ocorrência", y=1.08, fontsize=12)

        # Salvar
        for fmt in save_formats:
            out_path = os.path.join(output_dir, f"{basename}_{idx:02d}.{fmt}")
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            out_files.append(out_path)

        plt.close(fig)

    return out_files


# -----------------------------
# Pipeline completo (RAP-friendly)
# -----------------------------
def run_lca_pipeline(
    parquet_path: str = "data_SinanDengue_2019_treated_part.parquet",
    n_classes: int = 13,
    feature_cols: Optional[Sequence[str]] = None,
    id_cols: Optional[Sequence[str]] = None,
    drop_na_rows: bool = True,
    n_init: int = 5,
    max_iter: int = 500,
    random_state: int = 42,
    verbose: int = 0,
    radar_order_by: str = "variance",   # "input" | "variance" | "mean"
    radar_max_vars: Optional[int] = None,
    output_dir: str = "lca_outputs",
) -> dict:
    """
    Executa o fluxo completo:
     1) Lê Parquet
     2) Seleciona colunas binárias
     3) Ajusta LCA (StepMix)
     4) Calcula perfis de classe (probabilidade de '1' por variável)
     5) Gera gráficos radar por classe + exports CSV

    Returns (dict):
        {
          "model": StepMix,
          "class_profiles": DataFrame (classes x variáveis),
          "class_weights": Series (peso de cada classe),
          "assignments": Series (classe argmax por indivíduo),
          "radar_files": [lista de paths],
          "features_used": [lista de colunas]
        }
    """
    # 1) Leitura + features binárias
    df, binary_cols = read_parquet_binary(
        parquet_path=parquet_path,
        feature_cols=feature_cols,
        id_cols=id_cols,
        drop_na_rows=drop_na_rows,
    )

    X = df[binary_cols].copy()

    # 2) Ajuste do modelo
    model = fit_lca_bernoulli(
        X=X,
        n_classes=n_classes,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

    # 3) Posteriores, pesos e atribuições
    post = model.predict_proba(X.values)  # (n, k)
    class_weights = pd.Series(post.mean(axis=0), index=[f"Classe_{i+1}" for i in range(n_classes)], name="peso")
    assignments = pd.Series(post.argmax(axis=1) + 1, name="classe_predita")  # 1..k

    # 4) Perfis de classe
    class_profiles = class_profiles_from_posteriors(
        X=X,
        posteriors=post,
        feature_names=binary_cols,
    )

    # 5) Exports
    os.makedirs(output_dir, exist_ok=True)
    class_profiles.to_csv(os.path.join(output_dir, "class_profiles.csv"), index=True, encoding="utf-8-sig")
    class_weights.to_csv(os.path.join(output_dir, "class_weights.csv"), header=True, encoding="utf-8-sig")

    # 6) Plots radar
    radar_files = plot_radar_per_class(
        class_profiles=class_profiles,
        order_features_by=radar_order_by,
        max_vars=radar_max_vars,
        output_dir=output_dir,
        basename="radar_class",
        save_formats=("png",),  # adicione "pdf" se quiser
        dpi=150,
    )

    # Também pode exportar as atribuições indivíduo-a-classe, se houver id_cols
    if id_cols:
        out_assign = df[id_cols].copy()
        out_assign["classe_predita"] = assignments.values
        out_assign.to_csv(os.path.join(output_dir, "individual_assignments.csv"), index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "class_profiles": class_profiles,
        "class_weights": class_weights,
        "assignments": assignments,
        "radar_files": radar_files,
        "features_used": binary_cols,
    }


# -----------------------------
# Execução direta (exemplo)
# -----------------------------
if __name__ == "__main__":
    """
    Rode no terminal integrado do VS Code:
        python lca_radar.py
    Os resultados (CSVs e PNGs) ficarão na pasta ./lca_outputs
    """
    results = run_lca_pipeline(
        parquet_path="data_SinanDengue_2019_treated_part.parquet",  # conforme solicitado
        n_classes=13,               # fixo por padrão (ajuste se desejar)
        feature_cols=None,          # ou lista explícita de colunas binárias
        id_cols=None,               # ex.: ["id_paciente", "ano"]
        drop_na_rows=True,
        n_init=8,
        max_iter=600,
        random_state=42,
        verbose=0,
        radar_order_by="variance",  # "input" | "variance" | "mean"
        radar_max_vars=30,          # limite de variáveis por radar (opcional)
        output_dir="lca_outputs",
    )
    print("Concluído. Arquivos salvos em:", os.path.abspath("lca_outputs"))
    print("Classes e pesos:\n", results["class_weights"])
