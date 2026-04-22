from __future__ import annotations

from pathlib import Path


DEFAULT_YEARS = list(range(2015, 2026))
PROCESSOS_FILENAME = "27_processoslicitatorios-61218.csv"

OUTPUT_ROOT = Path("outputs")
SCHEMA_AUDIT_DIRNAME = "schema_audit"
UNIFIED_DIRNAME = "unified_base"
CORRELATION_DIRNAME = "correlations"

DEFAULT_SAMPLE_ROWS = 50
DEFAULT_SAMPLE_EXPORT_ROWS = 5000

MISSING_TOKENS = {
    "",
    " ",
    "  ",
    "   ",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "-",
}

TRUE_TOKENS = {"sim", "s", "true", "1", "yes", "y"}
FALSE_TOKENS = {"nao", "não", "nãƒo", "false", "0", "no", "n"}

COLUMN_ALIASES = {
    "nomefonecedor": "nome_fornecedor",
    "cnpjcpffornecedor": "cnpj_cpf_fornecedor",
    "cnpjcpfvencedor": "cnpj_cpf_vencedor",
    "cnpjcpfcontratado": "cnpj_cpf_contratado",
    "idfornecedor": "id_fornecedor",
    "nomeentidade": "nome_entidade",
    "numeroprocesso": "numero_processo",
    "anoprocesso": "ano_processo",
    "numerolicitacao": "numero_licitacao",
    "anolicitacao": "ano_licitacao",
    "datapublicacao": "data_publicacao",
    "datahomologacao": "data_homologacao",
    "dataaberturaenvelopes": "data_abertura_envelopes",
    "dataanulacao": "data_anulacao",
    "datacriacao": "data_criacao",
    "datajulgamento": "data_julgamento",
    "datarevogacao": "data_revogacao",
    "emailcontato": "email_contato",
    "enderecocertame": "endereco_certame",
    "enderecoentrega": "endereco_entrega",
    "estadocertame": "estado_certame",
    "formacontratacao": "forma_contratacao",
    "formajulgamento": "forma_julgamento",
    "iniciorecebimentoenvelopes": "inicio_recebimento_envelopes",
    "motivoanulacao": "motivo_anulacao",
    "motivorevogacao": "motivo_revogacao",
    "nomecontato": "nome_contato",
    "registroprecos": "registro_precos",
    "telefonecontato": "telefone_contato",
    "terminorecebimentoenvelopes": "termino_recebimento_envelopes",
    "fundamentolegal": "fundamento_legal",
    "valorestimado": "valor_estimado",
    "valorhomologado": "valor_homologado",
    "formaparticipacao": "forma_participacao",
    "itensproposta": "itens_proposta",
    "representantecertame": "representante_certame",
    "responsavelfornecedor": "responsavel_fornecedor",
    "unidademedida": "unidade_medida",
    "valortotal": "valor_total",
    "valorunitario": "valor_unitario",
    "iditem": "id_item",
    "descricaoelemento": "descricao_elemento",
    "fonterecurso": "fonte_recurso",
}

PROCESS_DATE_COLUMNS = [
    "data_publicacao",
    "data_homologacao",
    "data_abertura_envelopes",
    "data_anulacao",
    "data_criacao",
    "data_julgamento",
    "data_revogacao",
    "inicio_recebimento_envelopes",
    "termino_recebimento_envelopes",
]

PROCESS_NUMERIC_COLUMNS = [
    "numero_processo",
    "ano_processo",
    "numero_licitacao",
    "ano_licitacao",
    "valor_estimado",
    "valor_homologado",
]

PARTICIPANT_TEXT_COLUMNS = [
    "cnpj_cpf_fornecedor",
    "forma_participacao",
    "id_fornecedor",
    "itens_proposta",
    "nome_fornecedor",
    "representante_certame",
    "responsavel_fornecedor",
    "socios",
]

ITEM_NUMERIC_COLUMNS = [
    "id",
    "id_item",
    "numero",
    "codigo",
    "quantidade",
    "valor_total",
    "valor_unitario",
    "valor_total_referencia",
    "valor_unitario_referencia",
]

REFERENCE_COLUMNS = [
    "atas_registro_precos",
    "contratos",
    "despesas",
    "documentos_relacionados",
    "empenhos",
    "itens_vencedores",
    "participantes",
]

TEXT_COLUMNS_FOR_BASE = [
    "nome_entidade",
    "objeto",
    "modalidade",
    "tipo_objeto",
    "forma_julgamento",
    "situacao",
    "forma_contratacao",
    "meio_divulgacao",
    "fundamento_legal",
    "cnpj_cpf_fornecedor",
    "forma_participacao",
    "nome_fornecedor",
    "descricao_item",
    "unidade_medida_item",
    "situacao_item",
]

BASE_ID_COLUMNS = [
    "ano_dados",
    "licitacao_id",
    "participante_id",
    "item_proposta_id",
]

TARGET_COLUMN = "valor_homologado"
TARGET_DERIVED_FEATURES = [
    "diferenca_estimado_homologado",
    "desconto_relativo_estimado_homologado",
    "proporcao_valor_item_vs_homologado",
]

PRIMARY_FEATURE_CANDIDATES = [
    "valor_estimado",
    "quantidade_item",
    "valor_total_item",
    "valor_unitario_item",
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
    "descricao_item_num_chars",
    "descricao_item_num_palavras",
    "num_participantes_licitacao",
    "num_itens_proposta_participante",
    "num_itens_proposta_licitacao",
    "valor_estimado_por_participante",
    "valor_estimado_por_item_licitacao",
    "valor_total_item_por_participante",
    "valor_total_item_vs_estimado",
    "valor_unitario_item_vs_estimado",
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
    "item_venceu",
    "item_perdeu",
    "item_desclassificado",
    "item_inabilitado",
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

SECONDARY_DERIVED_FEATURES = [
    "log1p_valor_estimado",
    "log1p_quantidade_item",
    "log1p_valor_total_item",
    "log1p_valor_unitario_item",
    "log1p_num_participantes_licitacao",
    "log1p_num_itens_proposta_participante",
    "log1p_num_itens_proposta_licitacao",
    "log1p_valor_estimado_por_participante",
    "log1p_valor_estimado_por_item_licitacao",
    "sqrt_quantidade_item",
    "sqrt_num_participantes_licitacao",
    "sqrt_num_itens_proposta_participante",
    "sqrt_num_itens_proposta_licitacao",
    "densidade_texto_objeto",
    "densidade_texto_item",
    "valor_total_item_por_palavra_item",
    "valor_total_item_por_palavra_objeto",
]
