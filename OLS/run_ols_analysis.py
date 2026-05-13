from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from licitacoes_pipeline.config import OUTPUT_ROOT, TARGET_COLUMN, UNIFIED_DIRNAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calcula amostra e aplica OLS (numpy) com variáveis correlacionadas.")
    parser.add_argument("--input-path", type=Path, default=OUTPUT_ROOT / UNIFIED_DIRNAME / "licitacoes_unificadas.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("OLS") / "outputs")
    parser.add_argument("--target", type=str, default=TARGET_COLUMN)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--corr-threshold", type=float, default=0.2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--margin-error", type=float, default=0.05)
    parser.add_argument("--sample-cap", type=int, default=12000)
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_sample_size(population_size: int, confidence: float = 0.95, margin_error: float = 0.05, p: float = 0.5) -> int:
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(round(confidence, 2), 1.96)
    n0 = (z**2 * p * (1 - p)) / (margin_error**2)
    finite = n0 / (1 + ((n0 - 1) / max(population_size, 1)))
    return int(np.ceil(finite))


def fit_ols_numpy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    y_hat = X @ beta
    resid = y - y_hat

    sse = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (sse / sst if sst > 0 else 0.0)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

    sigma2 = sse / max(n - k, 1)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return beta, se, r2, r2_adj


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = pd.read_parquet(args.input_path)
    if args.target not in df.columns:
        raise ValueError(f"Variável alvo '{args.target}' não encontrada.")

    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df[numeric_df[args.target].notna()].copy()

    population_size = len(numeric_df)
    sample_size = min(compute_sample_size(population_size, args.confidence, args.margin_error), population_size)

    correlations = numeric_df.corr(numeric_only=True)[args.target].drop(labels=[args.target]).dropna()
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    selected = correlations[correlations.abs() >= args.corr_threshold].head(args.top_k)
    if selected.empty:
        selected = correlations.head(args.top_k)

    features = selected.index.tolist()
    model_df = numeric_df[[args.target] + features].dropna().copy()
    if len(model_df) > args.sample_cap:
        model_df = model_df.sample(n=args.sample_cap, random_state=42)

    X_raw = model_df[features].astype(float).to_numpy()
    y = model_df[args.target].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(X_raw)), X_raw])

    beta, se, r2, r2_adj = fit_ols_numpy(X, y)
    y_pred = X @ beta

    coef_df = pd.DataFrame({
        "feature": ["const"] + features,
        "coef": beta,
        "std_error": se,
        "coef_abs": np.abs(beta),
    }).sort_values("coef_abs", ascending=False)

    corr_df = pd.DataFrame({"feature": selected.index, "corr_target": selected.values, "corr_abs": selected.abs().values})
    pred_df = pd.DataFrame({"y_real": y, "y_previsto": y_pred, "erro": y - y_pred})

    corr_df.to_csv(output_dir / "features_correlacionadas.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(output_dir / "coeficientes_ols.csv", index=False, encoding="utf-8-sig")
    pred_df.head(300).to_csv(output_dir / "amostra_real_vs_previsto.csv", index=False, encoding="utf-8-sig")

    resumo = [
        f"Arquivo de entrada: {args.input_path}",
        f"Variável alvo: {args.target}",
        f"População elegível (alvo não nulo): {population_size}",
        f"Amostra recomendada (Cochran, {int(args.confidence*100)}% confiança, erro {args.margin_error:.2f}): {sample_size}",
        f"Variáveis correlacionadas usadas no OLS: {len(features)}",
        f"Registros usados no OLS: {len(model_df)}",
        f"R²: {r2:.4f}",
        f"R² ajustado: {r2_adj:.4f}",
        "",
        "Top 10 coeficientes absolutos:",
    ]
    for row in coef_df.head(10).itertuples(index=False):
        resumo.append(f"- {row.feature}: coef={row.coef:.6f}, std_err={row.std_error:.6f}")

    (output_dir / "resumo_ols.txt").write_text("\n".join(resumo), encoding="utf-8")

    print("[OLS] Análise concluída")
    print((output_dir / "resumo_ols.txt").as_posix())
    print((output_dir / "features_correlacionadas.csv").as_posix())
    print((output_dir / "coeficientes_ols.csv").as_posix())
    print((output_dir / "amostra_real_vs_previsto.csv").as_posix())


if __name__ == "__main__":
    main()
