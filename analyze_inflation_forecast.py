from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from licitacoes_pipeline.config import OUTPUT_ROOT, TARGET_COLUMN, UNIFIED_DIRNAME
from licitacoes_pipeline.utils import ensure_directory


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "inflation_forecast"
DEFAULT_IPCA_PATH = Path("data/support/ipca_anual_2015_2025.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara custos de licitação com a inflação e gera uma previsão simples para 2026."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=OUTPUT_ROOT / UNIFIED_DIRNAME / "licitacoes_unificadas.parquet",
    )
    parser.add_argument("--ipca-path", type=Path, default=DEFAULT_IPCA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--id-column", default="licitacao_id")
    parser.add_argument("--year-column", default="ano_dados")
    parser.add_argument("--base-year", type=int, default=2025)
    parser.add_argument("--forecast-year", type=int, default=2026)
    return parser


def load_unique_licitacoes(
    input_path: Path,
    *,
    id_column: str,
    target_column: str,
    year_column: str,
) -> pd.DataFrame:
    columns = [id_column, target_column, year_column]
    df = pd.read_parquet(input_path, columns=columns)
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df[df[target_column].notna()].copy()
    df = df.sort_values([id_column, year_column]).drop_duplicates(subset=[id_column], keep="first")
    return df.reset_index(drop=True)


def load_ipca(ipca_path: Path) -> pd.DataFrame:
    ipca = pd.read_csv(ipca_path)
    ipca["ano"] = pd.to_numeric(ipca["ano"], errors="coerce").astype(int)
    ipca["ipca_percentual"] = pd.to_numeric(ipca["ipca_percentual"], errors="coerce")
    ipca["fator_ipca"] = pd.to_numeric(ipca["fator_ipca"], errors="coerce")
    return ipca.sort_values("ano").reset_index(drop=True)


def build_price_index(ipca: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    cumulative = 100.0
    for row in ipca.itertuples(index=False):
        cumulative *= 1.0 + float(row.fator_ipca)
        rows.append(
            {
                "ano": int(row.ano),
                "ipca_percentual": float(row.ipca_percentual),
                "fator_ipca": float(row.fator_ipca),
                "indice_precos": cumulative,
            }
        )
    return pd.DataFrame(rows)


def annual_summary(
    df: pd.DataFrame,
    *,
    year_column: str,
    target_column: str,
    real_column: str,
) -> pd.DataFrame:
    grouped = df.groupby(year_column)
    summary = grouped.agg(
        quantidade_licitacoes=(target_column, "count"),
        media_nominal=(target_column, "mean"),
        mediana_nominal=(target_column, "median"),
        variancia_nominal=(target_column, "var"),
        desvio_padrao_nominal=(target_column, "std"),
        media_real_2025=(real_column, "mean"),
        mediana_real_2025=(real_column, "median"),
        variancia_real_2025=(real_column, "var"),
        desvio_padrao_real_2025=(real_column, "std"),
    ).reset_index()
    return summary.sort_values(year_column).reset_index(drop=True)


def add_growth_comparison(summary: pd.DataFrame, ipca_index: pd.DataFrame, *, year_column: str) -> pd.DataFrame:
    result = summary.merge(ipca_index[["ano", "ipca_percentual", "indice_precos"]], left_on=year_column, right_on="ano", how="left")
    result = result.drop(columns=["ano"])

    for column in ["media_nominal", "mediana_nominal", "media_real_2025", "mediana_real_2025"]:
        result[f"{column}_crescimento_pct"] = result[column].pct_change() * 100.0

    result["gap_mediana_nominal_menos_ipca"] = result["mediana_nominal_crescimento_pct"] - result["ipca_percentual"]
    result["gap_media_nominal_menos_ipca"] = result["media_nominal_crescimento_pct"] - result["ipca_percentual"]
    return result


def fit_predict(train_years: np.ndarray, train_values: np.ndarray, target_year: int, model_name: str) -> float:
    if model_name == "naive_last":
        return float(train_values[-1])
    if model_name == "linear_level":
        coef = np.polyfit(train_years, train_values, 1)
        return float(np.polyval(coef, target_year))
    if model_name == "linear_log":
        coef = np.polyfit(train_years, np.log(train_values), 1)
        return float(np.exp(np.polyval(coef, target_year)))
    raise ValueError(f"Modelo desconhecido: {model_name}")


def evaluate_models(years: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    candidate_models = ["naive_last", "linear_level", "linear_log"]
    rows: list[dict[str, float | str | int]] = []
    min_train_size = 5

    for model_name in candidate_models:
        predictions: list[float] = []
        actuals: list[float] = []
        target_years: list[int] = []
        for index in range(min_train_size, len(years)):
            train_years = years[:index]
            train_values = values[:index]
            target_year = int(years[index])
            target_value = float(values[index])
            prediction = fit_predict(train_years, train_values, target_year, model_name)
            predictions.append(prediction)
            actuals.append(target_value)
            target_years.append(target_year)

        pred = np.asarray(predictions, dtype=float)
        act = np.asarray(actuals, dtype=float)
        mae = float(np.mean(np.abs(pred - act)))
        rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
        mape = float(np.mean(np.abs((pred - act) / act)) * 100.0)
        rows.append(
            {
                "modelo": model_name,
                "avaliacoes": len(target_years),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )

    return pd.DataFrame(rows).sort_values(["mape", "mae"]).reset_index(drop=True)


def forecast_next_year(
    annual_compare: pd.DataFrame,
    *,
    forecast_year: int,
    base_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = annual_compare["ano_dados"].to_numpy(dtype=float)
    median_real = annual_compare["mediana_real_2025"].to_numpy(dtype=float)
    mean_real = annual_compare["media_real_2025"].to_numpy(dtype=float)

    model_scores = evaluate_models(years, median_real)
    best_model = str(model_scores.iloc[0]["modelo"])

    median_real_forecast = fit_predict(years, median_real, forecast_year, best_model)
    mean_real_forecast = fit_predict(years, mean_real, forecast_year, best_model)

    ipca_2026_scenario = float(annual_compare["ipca_percentual"].tail(3).mean())
    last_row = annual_compare.loc[annual_compare["ano_dados"] == base_year].iloc[0]
    mediana_nominal_2026 = median_real_forecast * (1.0 + ipca_2026_scenario / 100.0)
    media_nominal_2026 = mean_real_forecast * (1.0 + ipca_2026_scenario / 100.0)

    forecast = pd.DataFrame(
        [
            {
                "ano_previsao": forecast_year,
                "modelo_escolhido_mediana_real": best_model,
                "ipca_cenario_2026_percentual": ipca_2026_scenario,
                "mediana_real_2025_prevista": median_real_forecast,
                "media_real_2025_prevista": mean_real_forecast,
                "mediana_nominal_prevista": mediana_nominal_2026,
                "media_nominal_prevista": media_nominal_2026,
                "variacao_real_mediana_vs_2025_pct": ((median_real_forecast / float(last_row["mediana_real_2025"])) - 1.0) * 100.0,
                "variacao_nominal_mediana_vs_2025_pct": ((mediana_nominal_2026 / float(last_row["mediana_nominal"])) - 1.0) * 100.0,
                "variacao_real_media_vs_2025_pct": ((mean_real_forecast / float(last_row["media_real_2025"])) - 1.0) * 100.0,
                "variacao_nominal_media_vs_2025_pct": ((media_nominal_2026 / float(last_row["media_nominal"])) - 1.0) * 100.0,
            }
        ]
    )
    return model_scores, forecast


def build_summary_text(
    *,
    annual_compare: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_scores: pd.DataFrame,
) -> str:
    first_year = annual_compare.iloc[0]
    last_year = annual_compare.iloc[-1]
    forecast = forecast_df.iloc[0]

    cumulative_nominal_median = ((float(last_year["mediana_nominal"]) / float(first_year["mediana_nominal"])) - 1.0) * 100.0
    cumulative_real_median = ((float(last_year["mediana_real_2025"]) / float(first_year["mediana_real_2025"])) - 1.0) * 100.0

    lines = [
        "Comparação entre custo de licitação e inflação",
        f"Período analisado: {int(first_year['ano_dados'])} a {int(last_year['ano_dados'])}",
        f"Mediana nominal 2015: {float(first_year['mediana_nominal']):.2f}",
        f"Mediana nominal 2025: {float(last_year['mediana_nominal']):.2f}",
        f"Crescimento acumulado da mediana nominal: {cumulative_nominal_median:.2f}%",
        f"Mediana real (base 2025) em 2015: {float(first_year['mediana_real_2025']):.2f}",
        f"Mediana real (base 2025) em 2025: {float(last_year['mediana_real_2025']):.2f}",
        f"Crescimento acumulado da mediana real: {cumulative_real_median:.2f}%",
        "",
        "Leitura:",
        "A comparação nominal mostra o aumento observado nos valores pagos ao longo do tempo.",
        "A comparação real remove o efeito do IPCA e mostra se houve aumento acima da inflação.",
        "Como a distribuição do custo é altamente assimétrica, a mediana anual foi tratada como a medida principal.",
        "",
        "Modelos testados para prever a mediana real de 2026:",
    ]

    for row in model_scores.itertuples(index=False):
        lines.append(f"- {row.modelo}: MAPE={row.mape:.2f}%, MAE={row.mae:.2f}, RMSE={row.rmse:.2f}")

    lines.extend(
        [
            "",
            "Previsão para 2026:",
            f"- Modelo escolhido: {forecast['modelo_escolhido_mediana_real']}",
            f"- Cenário de inflação 2026 adotado: {forecast['ipca_cenario_2026_percentual']:.2f}% (média 2023-2025)",
            f"- Mediana real prevista (base 2025): {forecast['mediana_real_2025_prevista']:.2f}",
            f"- Mediana nominal prevista: {forecast['mediana_nominal_prevista']:.2f}",
            f"- Variação real da mediana vs 2025: {forecast['variacao_real_mediana_vs_2025_pct']:.2f}%",
            f"- Variação nominal da mediana vs 2025: {forecast['variacao_nominal_mediana_vs_2025_pct']:.2f}%",
            f"- Erro médio percentual do modelo escolhido (MAPE): {float(model_scores.iloc[0]['mape']):.2f}%",
            "",
            "Interpretação prática:",
            "Se a mediana nominal projetada crescer mais do que a inflação esperada, isso sugere pressão real sobre o custo das licitações.",
            "Esse resultado pode servir como base inicial para planejamento orçamentário de 2026, mas não substitui uma análise econômica mais detalhada por categoria de compra.",
            "Como o erro histórico do modelo ainda é relevante, a previsão deve ser tratada como exploratória e não como valor definitivo de orçamento.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_directory(args.output_dir)

    lic_df = load_unique_licitacoes(
        args.input_path,
        id_column=args.id_column,
        target_column=args.target_column,
        year_column=args.year_column,
    )
    ipca = load_ipca(args.ipca_path)
    ipca_index = build_price_index(ipca)

    base_index = float(ipca_index.loc[ipca_index["ano"] == args.base_year, "indice_precos"].iloc[0])
    ipca_index["deflator_para_base"] = base_index / ipca_index["indice_precos"]

    lic_df = lic_df.merge(ipca_index[["ano", "ipca_percentual", "indice_precos", "deflator_para_base"]], left_on=args.year_column, right_on="ano", how="left")
    lic_df["valor_homologado_real_2025"] = lic_df[args.target_column] * lic_df["deflator_para_base"]
    lic_df = lic_df.drop(columns=["ano"])

    annual = annual_summary(
        lic_df,
        year_column=args.year_column,
        target_column=args.target_column,
        real_column="valor_homologado_real_2025",
    )
    annual_compare = add_growth_comparison(annual, ipca_index, year_column=args.year_column)
    model_scores, forecast_df = forecast_next_year(
        annual_compare,
        forecast_year=args.forecast_year,
        base_year=args.base_year,
    )

    detailed_path = output_dir / "licitacoes_deflacionadas.parquet"
    annual_path = output_dir / "custos_anuais_licitacao.csv"
    compare_path = output_dir / "comparativo_inflacao_custos.csv"
    model_path = output_dir / "avaliacao_modelos_previsao_2026.csv"
    forecast_path = output_dir / "previsao_2026.csv"
    summary_path = output_dir / "resumo_inflacao_previsao.txt"

    lic_df.to_parquet(detailed_path, index=False)
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")
    annual_compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    model_scores.to_csv(model_path, index=False, encoding="utf-8-sig")
    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")

    summary_text = build_summary_text(
        annual_compare=annual_compare,
        forecast_df=forecast_df,
        model_scores=model_scores,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"[analyze_inflation_forecast] Licitações deflacionadas: {detailed_path}")
    print(f"[analyze_inflation_forecast] Custos anuais: {annual_path}")
    print(f"[analyze_inflation_forecast] Comparativo inflação x custos: {compare_path}")
    print(f"[analyze_inflation_forecast] Avaliação dos modelos: {model_path}")
    print(f"[analyze_inflation_forecast] Previsão 2026: {forecast_path}")
    print(f"[analyze_inflation_forecast] Resumo: {summary_path}")


if __name__ == "__main__":
    main()
