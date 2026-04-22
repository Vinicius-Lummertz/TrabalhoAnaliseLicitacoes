from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from licitacoes_pipeline.config import (
    CORRELATION_DIRNAME,
    OUTPUT_ROOT,
    PRIMARY_FEATURE_CANDIDATES,
    SECONDARY_DERIVED_FEATURES,
    TARGET_COLUMN,
    TARGET_DERIVED_FEATURES,
    UNIFIED_DIRNAME,
)
from licitacoes_pipeline.utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calcula correlações de Pearson com a variável alvo.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=OUTPUT_ROOT / UNIFIED_DIRNAME / "licitacoes_unificadas.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / CORRELATION_DIRNAME)
    parser.add_argument("--min-qualified", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.3)
    return parser


def add_secondary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "valor_estimado" in df.columns:
        df["log1p_valor_estimado"] = np.log1p(df["valor_estimado"].clip(lower=0))
    if "valor_estimado_por_participante" in df.columns:
        df["log1p_valor_estimado_por_participante"] = np.log1p(df["valor_estimado_por_participante"].clip(lower=0))
    if "valor_estimado_por_item_licitacao" in df.columns:
        df["log1p_valor_estimado_por_item_licitacao"] = np.log1p(df["valor_estimado_por_item_licitacao"].clip(lower=0))
    if "quantidade_item" in df.columns:
        df["log1p_quantidade_item"] = np.log1p(df["quantidade_item"].clip(lower=0))
        df["sqrt_quantidade_item"] = np.sqrt(df["quantidade_item"].clip(lower=0))
    if "valor_total_item" in df.columns:
        df["log1p_valor_total_item"] = np.log1p(df["valor_total_item"].clip(lower=0))
    if "valor_unitario_item" in df.columns:
        df["log1p_valor_unitario_item"] = np.log1p(df["valor_unitario_item"].clip(lower=0))
    if "num_participantes_licitacao" in df.columns:
        df["log1p_num_participantes_licitacao"] = np.log1p(df["num_participantes_licitacao"].clip(lower=0))
        df["sqrt_num_participantes_licitacao"] = np.sqrt(df["num_participantes_licitacao"].clip(lower=0))
    if "num_itens_proposta_participante" in df.columns:
        df["log1p_num_itens_proposta_participante"] = np.log1p(df["num_itens_proposta_participante"].clip(lower=0))
        df["sqrt_num_itens_proposta_participante"] = np.sqrt(df["num_itens_proposta_participante"].clip(lower=0))
    if "num_itens_proposta_licitacao" in df.columns:
        df["log1p_num_itens_proposta_licitacao"] = np.log1p(df["num_itens_proposta_licitacao"].clip(lower=0))
        df["sqrt_num_itens_proposta_licitacao"] = np.sqrt(df["num_itens_proposta_licitacao"].clip(lower=0))
    if {"objeto_num_chars", "objeto_num_palavras"}.issubset(df.columns):
        df["densidade_texto_objeto"] = df["objeto_num_chars"] / df["objeto_num_palavras"].replace(0, pd.NA)
        df["valor_total_item_por_palavra_objeto"] = df["valor_total_item"] / df["objeto_num_palavras"].replace(0, pd.NA)
    if {"descricao_item_num_chars", "descricao_item_num_palavras"}.issubset(df.columns):
        df["densidade_texto_item"] = df["descricao_item_num_chars"] / df["descricao_item_num_palavras"].replace(0, pd.NA)
        df["valor_total_item_por_palavra_item"] = df["valor_total_item"] / df["descricao_item_num_palavras"].replace(0, pd.NA)
    return df


def compute_correlations(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        subset = df[[feature, target]].dropna()
        if len(subset) < 3:
            continue
        if subset[feature].nunique(dropna=True) <= 1:
            continue
        correlation, pvalue = pearsonr(subset[feature], subset[target])
        rows.append(
            {
                "variavel": feature,
                "correlacao_pearson": float(correlation),
                "correlacao_absoluta": float(abs(correlation)),
                "p_valor": float(pvalue),
                "amostra_utilizada": int(len(subset)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("correlacao_absoluta", ascending=False).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_directory(args.output_dir)

    df = pd.read_parquet(args.input_path)
    df = df[df[TARGET_COLUMN].notna()].copy()

    excluded_prefixes = ("licitacao_id", "participante_id", "item_proposta_id")
    numeric_candidates = []
    for column in df.columns:
        if column == TARGET_COLUMN:
            continue
        if column in TARGET_DERIVED_FEATURES:
            continue
        if any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_bool_dtype(df[column]) or pd.api.types.is_numeric_dtype(df[column]):
            numeric_candidates.append(column)

    primary = [feature for feature in PRIMARY_FEATURE_CANDIDATES if feature in numeric_candidates]
    correlations = compute_correlations(df, primary, TARGET_COLUMN)
    qualified = correlations[correlations["correlacao_absoluta"] >= args.threshold]

    rerun_used = False
    if len(qualified) < args.min_qualified:
        rerun_used = True
        df = add_secondary_features(df)
        secondary = [feature for feature in SECONDARY_DERIVED_FEATURES if feature in df.columns]
        all_features = list(dict.fromkeys(primary + secondary + numeric_candidates))
        correlations = compute_correlations(df, all_features, TARGET_COLUMN)
        qualified = correlations[correlations["correlacao_absoluta"] >= args.threshold]

    correlations_path = output_dir / "correlacoes_completas.csv"
    qualified_path = output_dir / "correlacoes_filtradas.csv"
    matrix_path = output_dir / "matriz_correlacao_final.csv"
    summary_path = output_dir / "resumo_correlacoes.txt"

    correlations.to_csv(correlations_path, index=False, encoding="utf-8-sig")
    qualified.to_csv(qualified_path, index=False, encoding="utf-8-sig")

    final_features = [TARGET_COLUMN] + qualified["variavel"].head(max(args.min_qualified, len(qualified))).tolist()
    final_features = list(dict.fromkeys(final_features))
    matrix = df[final_features].corr(method="pearson")
    matrix.to_csv(matrix_path, encoding="utf-8-sig")

    summary_lines = [
        f"Arquivo de entrada: {args.input_path}",
        f"Registros válidos para a análise: {len(df)}",
        f"Variáveis primárias avaliadas: {len(primary)}",
        f"Variáveis aprovadas com |r| >= {args.threshold}: {len(qualified)}",
        f"Nova rodada de features derivadas executada: {'sim' if rerun_used else 'não'}",
        "Top 10 correlações:",
    ]
    for row in correlations.head(10).itertuples(index=False):
        summary_lines.append(f"- {row.variavel}: r={row.correlacao_pearson:.4f}, p={row.p_valor:.4e}, n={row.amostra_utilizada}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[analyze_correlations] Correlações completas: {correlations_path}")
    print(f"[analyze_correlations] Correlações filtradas: {qualified_path}")
    print(f"[analyze_correlations] Matriz final: {matrix_path}")
    print(f"[analyze_correlations] Resumo: {summary_path}")


if __name__ == "__main__":
    main()
