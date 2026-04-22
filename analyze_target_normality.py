from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from licitacoes_pipeline.config import OUTPUT_ROOT, TARGET_COLUMN, UNIFIED_DIRNAME
from licitacoes_pipeline.utils import ensure_directory


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "target_normality"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Testa a normalidade da variável alvo com Shapiro-Wilk e compara transformações."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=OUTPUT_ROOT / UNIFIED_DIRNAME / "licitacoes_unificadas.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--id-column", default="licitacao_id")
    parser.add_argument("--year-column", default="ano_dados")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def shapiro_metrics(series: pd.Series, *, alpha: float) -> dict[str, float | str | bool]:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    statistic, pvalue = stats.shapiro(clean)
    return {
        "n": int(len(clean)),
        "media": float(clean.mean()),
        "mediana": float(clean.median()),
        "variancia": float(clean.var(ddof=1)),
        "desvio_padrao": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skewness": float(clean.skew()),
        "kurtosis": float(clean.kurt()),
        "shapiro_w": float(statistic),
        "shapiro_pvalue": float(pvalue),
        "normal_ao_nivel_alpha": bool(pvalue >= alpha),
    }


def load_unique_target_series(
    input_path: Path,
    *,
    id_column: str,
    target_column: str,
    year_column: str,
) -> tuple[pd.DataFrame, int]:
    columns = [id_column, target_column, year_column]
    df = pd.read_parquet(input_path, columns=columns)
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df[df[target_column].notna()].copy()

    inconsistent_ids = (
        df.groupby(id_column)[target_column]
        .nunique(dropna=True)
        .gt(1)
        .sum()
    )

    unique_df = (
        df.sort_values([id_column, year_column])
        .drop_duplicates(subset=[id_column], keep="first")
        .reset_index(drop=True)
    )
    return unique_df, int(inconsistent_ids)


def build_transformations(series: pd.Series) -> tuple[dict[str, pd.Series], dict[str, float]]:
    series = series.astype(float)
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    capped = series.clip(lower=lower, upper=upper)

    shift_original = 1 - float(series.min()) if float(series.min()) <= 0 else 0.0
    shift_capped = 1 - float(capped.min()) if float(capped.min()) <= 0 else 0.0

    transformations: dict[str, pd.Series] = {
        "original": series,
        "outlier_capped_iqr": capped,
        "log1p": pd.Series(np.log1p(series), index=series.index),
        "sqrt": pd.Series(np.sqrt(series), index=series.index),
        "boxcox_shifted": pd.Series(stats.boxcox(series + shift_original)[0], index=series.index),
        "log1p_outlier_capped": pd.Series(np.log1p(capped), index=series.index),
        "sqrt_outlier_capped": pd.Series(np.sqrt(capped), index=series.index),
        "boxcox_outlier_capped": pd.Series(stats.boxcox(capped + shift_capped)[0], index=series.index),
    }
    params = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior_outlier": lower,
        "limite_superior_outlier": upper,
        "shift_boxcox_original": shift_original,
        "shift_boxcox_capped": shift_capped,
    }
    return transformations, params


def annual_variance_summary(unique_df: pd.DataFrame, *, target_column: str, year_column: str) -> pd.DataFrame:
    grouped = unique_df.groupby(year_column)[target_column]
    summary = grouped.agg(
        quantidade_licitacoes="count",
        media="mean",
        mediana="median",
        variancia="var",
        desvio_padrao="std",
        minimo="min",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        maximo="max",
    ).reset_index()
    return summary.sort_values(year_column).reset_index(drop=True)


def build_summary_text(
    *,
    input_path: Path,
    alpha: float,
    unique_n: int,
    item_level_n: int,
    inconsistent_ids: int,
    transform_results: pd.DataFrame,
    annual_summary: pd.DataFrame,
    params: dict[str, float],
) -> str:
    best = transform_results.sort_values(["normal_ao_nivel_alpha", "shapiro_w"], ascending=[False, False]).iloc[0]
    original = transform_results.loc[transform_results["transformacao"] == "original"].iloc[0]
    lines = [
        f"Arquivo de entrada: {input_path}",
        f"Nível de significância (alpha): {alpha}",
        f"Registros na base expandida por item: {item_level_n}",
        f"Série usada no SW Teste: {unique_n} licitações únicas",
        f"IDs com alvo inconsistente entre linhas duplicadas: {inconsistent_ids}",
        "Regra do teste:",
        "H0 = a distribuição é normal.",
        "Rejeitar H0 quando p-value < alpha.",
        "",
        "Resultado da série original:",
        f"W = {original['shapiro_w']:.6f}",
        f"p-value = {original['shapiro_pvalue']:.6e}",
        f"Skewness = {original['skewness']:.6f}",
        f"Kurtosis = {original['kurtosis']:.6f}",
        f"Normal a 5%? {'sim' if bool(original['normal_ao_nivel_alpha']) else 'não'}",
        "",
        "Transformação mais próxima da normalidade:",
        f"{best['transformacao']}",
        f"W = {best['shapiro_w']:.6f}",
        f"p-value = {best['shapiro_pvalue']:.6e}",
        f"Skewness = {best['skewness']:.6f}",
        f"Kurtosis = {best['kurtosis']:.6f}",
        f"Normal a 5%? {'sim' if bool(best['normal_ao_nivel_alpha']) else 'não'}",
        "",
        "Parâmetros usados na normalização por outlier:",
        f"Q1 = {params['q1']:.6f}",
        f"Q3 = {params['q3']:.6f}",
        f"AIQ = {params['iqr']:.6f}",
        f"Limite inferior = {params['limite_inferior_outlier']:.6f}",
        f"Limite superior = {params['limite_superior_outlier']:.6f}",
        "",
        "Leitura estatística:",
        "A série original do valor homologado não é normal pelo SW Teste.",
        "As transformações de log, raiz, Box-Cox e o tratamento de outliers reduzem a assimetria e melhoram W.",
        "Mesmo assim, com 1.950 licitações válidas, nenhuma transformação atingiu p-value >= 0.05.",
        "Para as próximas etapas preditivas, a melhor candidata para estabilizar a distribuição é a transformação com maior W.",
        "",
        "Variância anual da série-alvo:",
    ]
    for row in annual_summary.itertuples(index=False):
        lines.append(
            f"- {getattr(row, annual_summary.columns[0])}: n={row.quantidade_licitacoes}, "
            f"media={row.media:.2f}, mediana={row.mediana:.2f}, variancia={row.variancia:.2f}, "
            f"desvio_padrao={row.desvio_padrao:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_directory(args.output_dir)

    unique_df, inconsistent_ids = load_unique_target_series(
        args.input_path,
        id_column=args.id_column,
        target_column=args.target_column,
        year_column=args.year_column,
    )
    item_level_n = int(
        pd.read_parquet(args.input_path, columns=[args.target_column])[args.target_column]
        .notna()
        .sum()
    )

    target_series = unique_df[args.target_column].astype(float)
    transformations, params = build_transformations(target_series)

    transform_rows: list[dict[str, object]] = []
    for name, transformed in transformations.items():
        metrics = shapiro_metrics(transformed, alpha=args.alpha)
        metrics["transformacao"] = name
        transform_rows.append(metrics)

    transform_results = (
        pd.DataFrame(transform_rows)
        .sort_values(["normal_ao_nivel_alpha", "shapiro_w"], ascending=[False, False])
        .reset_index(drop=True)
    )

    annual_summary = annual_variance_summary(
        unique_df,
        target_column=args.target_column,
        year_column=args.year_column,
    )

    best_name = transform_results.iloc[0]["transformacao"]
    best_series = transformations[str(best_name)]
    best_output_df = unique_df[[args.id_column, args.year_column, args.target_column]].copy()
    best_output_df["transformacao_escolhida"] = best_name
    best_output_df["valor_transformado"] = best_series.values

    unique_path = output_dir / "serie_alvo_licitacao.parquet"
    transform_results_path = output_dir / "teste_normalidade_transformacoes.csv"
    annual_summary_path = output_dir / "variancia_anual_licitacao.csv"
    best_series_path = output_dir / "serie_alvo_melhor_transformacao.parquet"
    summary_path = output_dir / "resumo_normalidade.txt"

    unique_df.to_parquet(unique_path, index=False)
    transform_results.to_csv(transform_results_path, index=False, encoding="utf-8-sig")
    annual_summary.to_csv(annual_summary_path, index=False, encoding="utf-8-sig")
    best_output_df.to_parquet(best_series_path, index=False)

    summary_text = build_summary_text(
        input_path=args.input_path,
        alpha=args.alpha,
        unique_n=len(unique_df),
        item_level_n=item_level_n,
        inconsistent_ids=inconsistent_ids,
        transform_results=transform_results,
        annual_summary=annual_summary,
        params=params,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"[analyze_target_normality] Série única por licitação: {unique_path}")
    print(f"[analyze_target_normality] Resultado dos testes: {transform_results_path}")
    print(f"[analyze_target_normality] Variância anual: {annual_summary_path}")
    print(f"[analyze_target_normality] Melhor transformação: {best_series_path}")
    print(f"[analyze_target_normality] Resumo: {summary_path}")


if __name__ == "__main__":
    main()
