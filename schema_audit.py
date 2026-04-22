from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from licitacoes_pipeline.config import DEFAULT_SAMPLE_ROWS, DEFAULT_YEARS, OUTPUT_ROOT, SCHEMA_AUDIT_DIRNAME
from licitacoes_pipeline.utils import (
    detect_encoding,
    detect_file_type,
    ensure_directory,
    header_signature,
    infer_column_kinds,
    read_csv_with_detection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audita os esquemas dos CSVs por ano e tipo.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / SCHEMA_AUDIT_DIRNAME)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--years", nargs="*", type=int, default=DEFAULT_YEARS)
    return parser


def audit_files(data_dir: Path, years: list[int], sample_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        print(f"[schema_audit] Auditando {year_dir}...")
        for path in sorted(year_dir.glob("*.csv")):
            preview = read_csv_with_detection(path, nrows=sample_rows)
            columns = list(preview.columns)
            detail_rows.append(
                {
                    "ano_dados": year,
                    "arquivo": path.name,
                    "caminho": str(path),
                    "tipo_arquivo": detect_file_type(path),
                    "encoding_detectado": detect_encoding(path),
                    "qtd_colunas": len(columns),
                    "colunas_json": json.dumps(columns, ensure_ascii=False),
                    "tipos_inferidos_json": json.dumps(infer_column_kinds(preview), ensure_ascii=False),
                    "header_hash": header_signature(columns),
                    "linhas_amostra": len(preview),
                }
            )

    details = pd.DataFrame(detail_rows)
    if details.empty:
        return details, pd.DataFrame()

    summary = (
        details.groupby(["ano_dados", "tipo_arquivo", "header_hash"], dropna=False)
        .agg(
            qtd_arquivos=("arquivo", "count"),
            qtd_colunas=("qtd_colunas", "first"),
            encoding_mais_comum=("encoding_detectado", lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]),
            colunas_json=("colunas_json", "first"),
            tipos_inferidos_json=("tipos_inferidos_json", "first"),
        )
        .reset_index()
        .sort_values(["ano_dados", "tipo_arquivo", "qtd_arquivos"], ascending=[True, True, False])
    )
    return details, summary


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_directory(args.output_dir)
    details, summary = audit_files(args.data_dir, args.years, args.sample_rows)

    detail_path = output_dir / "schema_audit_detalhado.csv"
    summary_path = output_dir / "schema_audit_resumo.csv"
    if not details.empty:
        details.to_csv(detail_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[schema_audit] Arquivo detalhado: {detail_path}")
    print(f"[schema_audit] Arquivo resumo: {summary_path}")


if __name__ == "__main__":
    main()

