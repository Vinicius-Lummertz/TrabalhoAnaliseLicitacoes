from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from licitacoes_pipeline.config import (
    BASE_ID_COLUMNS,
    DEFAULT_SAMPLE_EXPORT_ROWS,
    DEFAULT_YEARS,
    ITEM_NUMERIC_COLUMNS,
    OUTPUT_ROOT,
    PROCESSOS_FILENAME,
    PROCESS_DATE_COLUMNS,
    PROCESS_NUMERIC_COLUMNS,
    REFERENCE_COLUMNS,
    TEXT_COLUMNS_FOR_BASE,
    UNIFIED_DIRNAME,
)
from licitacoes_pipeline.utils import (
    boolean_to_int_series,
    concat_frames,
    ensure_directory,
    has_reference,
    normalize_columns,
    normalize_string_series,
    normalize_token,
    parse_boolean,
    parse_datetime_series,
    parse_numeric_series,
    read_csv_with_detection,
    stable_hash,
    text_length,
    word_count,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera a base analítica unificada de licitações.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / UNIFIED_DIRNAME)
    parser.add_argument("--years", nargs="*", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_EXPORT_ROWS)
    return parser


def safe_ref(base_dir: Path, value: object) -> Path | None:
    if not has_reference(value):
        return None
    path = base_dir / str(value).strip()
    return path if path.exists() else None


def build_process_frame(process_path: Path, year: int) -> pd.DataFrame:
    process_df = normalize_columns(read_csv_with_detection(process_path))
    process_df["ano_dados"] = year

    for column in PROCESS_DATE_COLUMNS:
        if column in process_df.columns:
            process_df[column] = parse_datetime_series(process_df[column])

    for column in PROCESS_NUMERIC_COLUMNS:
        if column in process_df.columns:
            process_df[column] = parse_numeric_series(process_df[column])

    text_columns = set(process_df.columns) - set(PROCESS_DATE_COLUMNS) - set(PROCESS_NUMERIC_COLUMNS) - {"ano_dados"}
    for column in text_columns:
        process_df[column] = normalize_string_series(process_df[column])

    process_df["licitacao_id"] = process_df.apply(
        lambda row: stable_hash(
            row.get("ano_processo"),
            row.get("numero_processo"),
            row.get("ano_licitacao"),
            row.get("numero_licitacao"),
            row.get("nome_entidade"),
            prefix="lic",
        ),
        axis=1,
    )
    process_df["arquivo_processo"] = process_path.name

    for ref_column in REFERENCE_COLUMNS:
        if ref_column not in process_df.columns:
            process_df[ref_column] = pd.Series(pd.NA, index=process_df.index, dtype="string")

    process_df["has_contrato"] = process_df["contratos"].map(has_reference).astype("int8")
    process_df["has_despesa"] = process_df["despesas"].map(has_reference).astype("int8")
    process_df["has_ata_registro_precos"] = process_df["atas_registro_precos"].map(has_reference).astype("int8")
    process_df["has_documentos_relacionados"] = process_df["documentos_relacionados"].map(has_reference).astype("int8")
    process_df["has_empenhos"] = process_df["empenhos"].map(has_reference).astype("int8")
    process_df["has_itens_vencedores"] = process_df["itens_vencedores"].map(has_reference).astype("int8")
    process_df["has_participantes"] = process_df["participantes"].map(has_reference).astype("int8")
    process_df["registro_precos_bool"] = boolean_to_int_series(
        process_df.get("registro_precos", pd.Series(pd.NA, index=process_df.index))
    )
    process_df["adesao_bool"] = boolean_to_int_series(
        process_df.get("adesao", pd.Series(pd.NA, index=process_df.index))
    )

    situacao_norm = process_df.get("situacao", pd.Series("", index=process_df.index)).fillna("").map(normalize_token)
    modalidade_norm = process_df.get("modalidade", pd.Series("", index=process_df.index)).fillna("").map(normalize_token)
    meio_norm = process_df.get("meio_divulgacao", pd.Series("", index=process_df.index)).fillna("").map(normalize_token)
    julgamento_norm = process_df.get("forma_julgamento", pd.Series("", index=process_df.index)).fillna("").map(normalize_token)
    forma_contratacao_norm = process_df.get("forma_contratacao", pd.Series("", index=process_df.index)).fillna("").map(normalize_token)

    process_df["situacao_licitacao_homologada"] = situacao_norm.str.contains("homolog", na=False).astype("int8")
    process_df["situacao_licitacao_revogada"] = situacao_norm.str.contains("revog", na=False).astype("int8")
    process_df["situacao_licitacao_anulada"] = situacao_norm.str.contains("anulad", na=False).astype("int8")
    process_df["is_pregao"] = modalidade_norm.str.contains("pregao", na=False).astype("int8")
    process_df["is_convite"] = modalidade_norm.str.contains("convite", na=False).astype("int8")
    process_df["is_concorrencia"] = modalidade_norm.str.contains("concorr", na=False).astype("int8")
    process_df["is_inexigibilidade"] = modalidade_norm.str.contains("inexig", na=False).astype("int8")
    process_df["is_dispensa"] = modalidade_norm.str.contains("dispensa", na=False).astype("int8")
    process_df["is_tomada_preco"] = modalidade_norm.str.contains("tomada", na=False).astype("int8")
    process_df["is_chamada_publica_credenciamento"] = forma_contratacao_norm.str.contains("chamada publica|credenciamento", na=False).astype("int8")
    process_df["meio_divulgacao_internet"] = meio_norm.str.contains("internet", na=False).astype("int8")
    process_df["meio_divulgacao_mural_publico"] = meio_norm.str.contains("mural", na=False).astype("int8")
    process_df["julgamento_menor_preco"] = julgamento_norm.str.contains("menor preco", na=False).astype("int8")

    process_df["ano"] = process_df["data_publicacao"].dt.year.astype("Int64")
    process_df["mes_publicacao"] = process_df["data_publicacao"].dt.month.astype("Int64")
    process_df["mes_homologacao"] = process_df["data_homologacao"].dt.month.astype("Int64")
    process_df["dia_publicacao"] = process_df["data_publicacao"].dt.day.astype("Int64")
    process_df["dia_homologacao"] = process_df["data_homologacao"].dt.day.astype("Int64")
    process_df["dias_publicacao_ate_homologacao"] = (
        process_df["data_homologacao"] - process_df["data_publicacao"]
    ).dt.days.astype("Float64")
    process_df["dias_publicacao_ate_abertura"] = (
        process_df["data_abertura_envelopes"] - process_df["data_publicacao"]
    ).dt.days.astype("Float64")
    process_df["prazo_recebimento_dias"] = (
        process_df["termino_recebimento_envelopes"] - process_df["inicio_recebimento_envelopes"]
    ).dt.days.astype("Float64")
    process_df["prazo_julgamento_apos_abertura"] = (
        process_df["data_julgamento"] - process_df["data_abertura_envelopes"]
    ).dt.days.astype("Float64")
    process_df["objeto_num_chars"] = text_length(process_df.get("objeto", pd.Series("", index=process_df.index)))
    process_df["objeto_num_palavras"] = word_count(process_df.get("objeto", pd.Series("", index=process_df.index)))
    process_df["diferenca_estimado_homologado"] = process_df["valor_estimado"] - process_df["valor_homologado"]
    process_df["desconto_relativo_estimado_homologado"] = (
        process_df["diferenca_estimado_homologado"] / process_df["valor_estimado"].replace(0, pd.NA)
    ).astype("Float64")

    return process_df


def build_base_for_year(year_dir: Path, year: int) -> tuple[pd.DataFrame, dict[str, int]]:
    process_path = year_dir / PROCESSOS_FILENAME
    process_df = build_process_frame(process_path, year)

    batch_frames: list[pd.DataFrame] = []
    year_frames: list[pd.DataFrame] = []
    stats = {
        "processos_lidos": int(len(process_df)),
        "processos_com_participantes": 0,
        "participantes_lidos": 0,
        "itens_lidos": 0,
        "participantes_sem_arquivo": 0,
        "itens_sem_arquivo": 0,
    }

    for process in process_df.itertuples(index=False):
        participant_path = safe_ref(year_dir, getattr(process, "participantes", None))
        if participant_path is None:
            stats["participantes_sem_arquivo"] += 1
            continue
        participants_df = normalize_columns(read_csv_with_detection(participant_path))
        if participants_df.empty:
            continue
        participant_payloads: list[tuple[int, pd.Series, pd.DataFrame, Path]] = []
        total_items_licitacao = 0

        for participant_idx, participant in participants_df.iterrows():
            participant_row = participant.copy()
            for column in participants_df.columns:
                participant_row[column] = normalize_string_series(pd.Series([participant_row[column]])).iat[0]

            item_path = safe_ref(year_dir, participant_row.get("itens_proposta"))
            if item_path is None:
                stats["itens_sem_arquivo"] += 1
                continue

            items_df = normalize_columns(read_csv_with_detection(item_path))
            if items_df.empty:
                continue

            for column in items_df.columns:
                if column in ITEM_NUMERIC_COLUMNS:
                    items_df[column] = parse_numeric_series(items_df[column])
                else:
                    items_df[column] = normalize_string_series(items_df[column])

            participant_payloads.append((participant_idx, participant_row, items_df, item_path))
            total_items_licitacao += int(len(items_df))

        if not participant_payloads:
            continue

        stats["processos_com_participantes"] += 1
        stats["participantes_lidos"] += int(len(participant_payloads))
        stats["itens_lidos"] += total_items_licitacao
        num_participantes = int(len(participant_payloads))

        for participant_idx, participant_row, items_df, item_path in participant_payloads:
            participante_id = stable_hash(
                getattr(process, "licitacao_id"),
                participant_row.get("id_fornecedor"),
                participant_row.get("cnpj_cpf_fornecedor"),
                participant_row.get("nome_fornecedor"),
                participant_idx,
                prefix="part",
            )
            num_itens_participante = int(len(items_df))

            items_df = items_df.rename(
                columns={
                    "descricao": "descricao_item",
                    "quantidade": "quantidade_item",
                    "unidade_medida": "unidade_medida_item",
                    "situacao": "situacao_item",
                    "valor_total": "valor_total_item",
                    "valor_unitario": "valor_unitario_item",
                }
            )

            original_item_id = items_df["id"] if "id" in items_df.columns else pd.Series(range(1, len(items_df) + 1), index=items_df.index)
            items_df["item_proposta_id"] = [
                stable_hash(
                    participante_id,
                    item_path.name,
                    original_item_id.iloc[idx],
                    idx,
                    prefix="item",
                )
                for idx in range(len(items_df))
            ]
            items_df["participante_id"] = participante_id
            items_df["licitacao_id"] = getattr(process, "licitacao_id")
            items_df["ano_dados"] = year
            items_df["arquivo_participantes"] = participant_path.name
            items_df["arquivo_itens_proposta"] = item_path.name
            items_df["arquivo_processo"] = getattr(process, "arquivo_processo")

            items_df["cnpj_cpf_fornecedor"] = participant_row.get("cnpj_cpf_fornecedor")
            items_df["forma_participacao"] = participant_row.get("forma_participacao")
            items_df["nome_fornecedor"] = participant_row.get("nome_fornecedor")
            items_df["id_fornecedor"] = participant_row.get("id_fornecedor")
            items_df["representante_certame"] = participant_row.get("representante_certame")
            items_df["responsavel_fornecedor"] = participant_row.get("responsavel_fornecedor")
            items_df["num_participantes_licitacao"] = num_participantes
            items_df["num_itens_proposta_participante"] = num_itens_participante

            items_df["nome_entidade"] = getattr(process, "nome_entidade", pd.NA)
            items_df["numero_processo"] = getattr(process, "numero_processo", pd.NA)
            items_df["ano_processo"] = getattr(process, "ano_processo", pd.NA)
            items_df["numero_licitacao"] = getattr(process, "numero_licitacao", pd.NA)
            items_df["ano_licitacao"] = getattr(process, "ano_licitacao", pd.NA)
            items_df["objeto"] = getattr(process, "objeto", pd.NA)
            items_df["modalidade"] = getattr(process, "modalidade", pd.NA)
            items_df["tipo_objeto"] = getattr(process, "tipo_objeto", pd.NA)
            items_df["forma_julgamento"] = getattr(process, "forma_julgamento", pd.NA)
            items_df["situacao"] = getattr(process, "situacao", pd.NA)
            items_df["forma_contratacao"] = getattr(process, "forma_contratacao", pd.NA)
            items_df["meio_divulgacao"] = getattr(process, "meio_divulgacao", pd.NA)
            items_df["fundamento_legal"] = getattr(process, "fundamento_legal", pd.NA)
            items_df["valor_estimado"] = getattr(process, "valor_estimado", pd.NA)
            items_df["valor_homologado"] = getattr(process, "valor_homologado", pd.NA)
            items_df["data_publicacao"] = getattr(process, "data_publicacao", pd.NaT)
            items_df["data_julgamento"] = getattr(process, "data_julgamento", pd.NaT)
            items_df["data_homologacao"] = getattr(process, "data_homologacao", pd.NaT)
            items_df["data_abertura_envelopes"] = getattr(process, "data_abertura_envelopes", pd.NaT)
            items_df["inicio_recebimento_envelopes"] = getattr(process, "inicio_recebimento_envelopes", pd.NaT)
            items_df["termino_recebimento_envelopes"] = getattr(process, "termino_recebimento_envelopes", pd.NaT)

            derived_from_process = [
                "ano",
                "mes_publicacao",
                "mes_homologacao",
                "dia_publicacao",
                "dia_homologacao",
                "dias_publicacao_ate_homologacao",
                "dias_publicacao_ate_abertura",
                "prazo_recebimento_dias",
                "prazo_julgamento_apos_abertura",
                "objeto_num_chars",
                "objeto_num_palavras",
                "diferenca_estimado_homologado",
                "desconto_relativo_estimado_homologado",
                "has_contrato",
                "has_despesa",
                "has_ata_registro_precos",
                "has_documentos_relacionados",
                "has_empenhos",
                "has_itens_vencedores",
                "has_participantes",
                "registro_precos_bool",
                "adesao_bool",
                "situacao_licitacao_homologada",
                "situacao_licitacao_revogada",
                "situacao_licitacao_anulada",
                "is_pregao",
                "is_convite",
                "is_concorrencia",
                "is_inexigibilidade",
                "is_dispensa",
                "is_tomada_preco",
                "is_chamada_publica_credenciamento",
                "meio_divulgacao_internet",
                "meio_divulgacao_mural_publico",
                "julgamento_menor_preco",
            ]
            for column in derived_from_process:
                items_df[column] = getattr(process, column, pd.NA)

            situacao_item_norm = items_df.get("situacao_item", pd.Series("", index=items_df.index)).fillna("").map(normalize_token)
            items_df["item_venceu"] = situacao_item_norm.str.contains("venceu", na=False).astype("int8")
            items_df["item_perdeu"] = situacao_item_norm.str.contains("perdeu", na=False).astype("int8")
            items_df["item_desclassificado"] = situacao_item_norm.str.contains("desclass", na=False).astype("int8")
            items_df["item_inabilitado"] = situacao_item_norm.str.contains("inabil", na=False).astype("int8")

            items_df["descricao_item_num_chars"] = text_length(items_df.get("descricao_item", pd.Series("", index=items_df.index)))
            items_df["descricao_item_num_palavras"] = word_count(items_df.get("descricao_item", pd.Series("", index=items_df.index)))

            grouped_total = items_df["valor_total_item"].sum(min_count=1)
            items_df["num_itens_proposta_licitacao"] = total_items_licitacao
            items_df["valor_total_item_por_participante"] = (
                items_df["valor_total_item"] / items_df["num_participantes_licitacao"].replace(0, pd.NA)
            ).astype("Float64")
            items_df["valor_total_item_vs_estimado"] = (
                items_df["valor_total_item"] / items_df["valor_estimado"].replace(0, pd.NA)
            ).astype("Float64")
            items_df["valor_unitario_item_vs_estimado"] = (
                items_df["valor_unitario_item"] / items_df["valor_estimado"].replace(0, pd.NA)
            ).astype("Float64")
            items_df["valor_estimado_por_participante"] = (
                items_df["valor_estimado"] / items_df["num_participantes_licitacao"].replace(0, pd.NA)
            ).astype("Float64")
            items_df["valor_estimado_por_item_licitacao"] = (
                items_df["valor_estimado"] / items_df["num_itens_proposta_licitacao"].replace(0, pd.NA)
            ).astype("Float64")
            items_df["proporcao_valor_item_vs_homologado"] = (
                items_df["valor_total_item"] / items_df["valor_homologado"].replace(0, pd.NA)
            ).astype("Float64")
            items_df["proporcao_valor_item_no_participante"] = (
                items_df["valor_total_item"] / grouped_total if pd.notna(grouped_total) and grouped_total != 0 else pd.NA
            )

            batch_frames.append(items_df)
            if len(batch_frames) >= 500:
                year_frames.append(concat_frames(batch_frames))
                batch_frames.clear()

    if batch_frames:
        year_frames.append(concat_frames(batch_frames))

    if year_frames:
        year_df = concat_frames(year_frames)
    else:
        year_df = pd.DataFrame()
    return year_df, stats


def finalize_base(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_columns = [
        "valor_estimado",
        "valor_homologado",
        "quantidade_item",
        "valor_total_item",
        "valor_unitario_item",
        "num_participantes_licitacao",
        "num_itens_proposta_participante",
        "num_itens_proposta_licitacao",
        "valor_total_item_por_participante",
        "valor_total_item_vs_estimado",
        "valor_unitario_item_vs_estimado",
        "valor_estimado_por_participante",
        "valor_estimado_por_item_licitacao",
        "proporcao_valor_item_vs_homologado",
        "proporcao_valor_item_no_participante",
        "descricao_item_num_chars",
        "descricao_item_num_palavras",
        "objeto_num_chars",
        "objeto_num_palavras",
        "ano",
        "mes_publicacao",
        "mes_homologacao",
        "dia_publicacao",
        "dia_homologacao",
        "dias_publicacao_ate_homologacao",
        "dias_publicacao_ate_abertura",
        "prazo_recebimento_dias",
        "prazo_julgamento_apos_abertura",
        "diferenca_estimado_homologado",
        "desconto_relativo_estimado_homologado",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in TEXT_COLUMNS_FOR_BASE:
        if column in df.columns:
            df[column] = normalize_string_series(df[column])

    ordered_columns = BASE_ID_COLUMNS + [
        "valor_homologado",
        "valor_estimado",
        "nome_entidade",
        "modalidade",
        "tipo_objeto",
        "forma_julgamento",
        "situacao",
        "forma_contratacao",
        "meio_divulgacao",
        "fundamento_legal",
        "data_publicacao",
        "data_julgamento",
        "data_homologacao",
        "data_abertura_envelopes",
        "inicio_recebimento_envelopes",
        "termino_recebimento_envelopes",
        "cnpj_cpf_fornecedor",
        "forma_participacao",
        "nome_fornecedor",
        "descricao_item",
        "quantidade_item",
        "unidade_medida_item",
        "situacao_item",
        "valor_total_item",
        "valor_unitario_item",
    ]
    remaining = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + remaining]


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_directory(args.output_dir)

    all_years: list[pd.DataFrame] = []
    stats_by_year: dict[str, dict[str, int]] = {}
    for year in args.years:
        year_dir = args.data_dir / str(year)
        if not year_dir.exists():
            continue
        print(f"[build_unified_base] Processando {year_dir}...")
        year_df, year_stats = build_base_for_year(year_dir, year)
        if not year_df.empty:
            all_years.append(year_df)
        stats_by_year[str(year)] = year_stats

    unified_df = finalize_base(concat_frames(all_years) if all_years else pd.DataFrame())
    parquet_path = output_dir / "licitacoes_unificadas.parquet"
    sample_path = output_dir / "licitacoes_unificadas_amostra.csv"
    quality_path = output_dir / "relatorio_qualidade.json"

    if not unified_df.empty:
        unified_df.to_parquet(parquet_path, index=False)
        unified_df.head(args.sample_rows).to_csv(sample_path, index=False, encoding="utf-8-sig")

    duplicates = 0
    if not unified_df.empty:
        duplicates = int(unified_df.duplicated(subset=["licitacao_id", "participante_id", "item_proposta_id"]).sum())

    quality_report = {
        "registros_finais": int(len(unified_df)),
        "anos_processados": args.years,
        "duplicidades_chave_primaria": duplicates,
        "registros_sem_valor_homologado": int(unified_df["valor_homologado"].isna().sum()) if "valor_homologado" in unified_df.columns else 0,
        "registros_sem_valor_estimado": int(unified_df["valor_estimado"].isna().sum()) if "valor_estimado" in unified_df.columns else 0,
        "registros_por_ano": unified_df["ano_dados"].value_counts(dropna=False).sort_index().to_dict() if "ano_dados" in unified_df.columns else {},
        "estatisticas_por_ano": stats_by_year,
    }
    write_json(quality_path, quality_report)

    print(f"[build_unified_base] Base final: {parquet_path}")
    print(f"[build_unified_base] Amostra CSV: {sample_path}")
    print(f"[build_unified_base] Relatório: {quality_path}")


if __name__ == "__main__":
    main()
