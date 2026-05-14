import os
import ast
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURAÇÕES
# =========================================================

PASTA_DATA = "data"
TARGET = "valorUnitarioVencedor"
ARQUIVO_BASE_FINAL = "base_final_analise.csv"
ARQUIVO_CORRELACOES = "todas_correlacoes.csv"
ARQUIVO_CORRELACOES_FORTES = "correlacoes_fortes.csv"

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def ler_csv_seguro(caminho_csv):
    encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                caminho_csv,
                encoding=encoding,
                sep=None,
                engine="python"
            )
        except Exception:
            continue

    return None


def converter_numero_br(valor):
    if pd.isna(valor):
        return np.nan

    valor = str(valor).strip()

    if valor == "":
        return np.nan

    valor = (
        valor
        .replace("R$", "")
        .replace(" ", "")
        .replace("\u00a0", "")
    )

    if "," in valor:
        valor = valor.replace(".", "").replace(",", ".")

    try:
        return float(valor)
    except Exception:
        return np.nan


def categorizar_objeto(texto):
    if pd.isna(texto):
        return "OUTROS"

    texto = str(texto).lower()

    if any(p in texto for p in ["paviment", "asfalt", "obra", "constru"]):
        return "OBRAS_E_INFRAESTRUTURA"

    if any(p in texto for p in ["aliment", "refei", "lanche", "jantar", "almoço", "café"]):
        return "ALIMENTACAO"

    if any(p in texto for p in ["transporte", "vale transporte", "bilhetagem"]):
        return "TRANSPORTE"

    if any(p in texto for p in ["medalha", "troféu", "premia"]):
        return "PREMIACAO"

    if any(p in texto for p in ["uniforme", "camiseta", "colete", "vestuario"]):
        return "VESTUARIO"

    if any(p in texto for p in ["kit", "material", "equipamento", "mobili", "biombo"]):
        return "MATERIAIS_E_EQUIPAMENTOS"

    if any(p in texto for p in ["serviço", "servico", "contratação", "prestação"]):
        return "SERVICOS"

    return "OUTROS"


def contar_lista_json(valor):
    if isinstance(valor, list):
        return len(valor)

    if pd.isna(valor):
        return 0

    texto = str(valor).strip()

    if texto == "":
        return 0

    try:
        convertido = ast.literal_eval(texto)
        if isinstance(convertido, list):
            return len(convertido)
    except Exception:
        pass

    return 0


# =========================================================
# 1. LEITURA E UNIFICAÇÃO DOS CSVs
# =========================================================

dfs = []

print("Lendo arquivos CSV...")

for ano in sorted(os.listdir(PASTA_DATA)):
    caminho_ano = os.path.join(PASTA_DATA, ano)

    if not os.path.isdir(caminho_ano):
        continue

    if not ano.isdigit():
        print(f"Ignorando pasta não anual: {ano}")
        continue

    for arquivo in os.listdir(caminho_ano):
        if not arquivo.lower().endswith(".csv"):
            continue

        caminho_csv = os.path.join(caminho_ano, arquivo)

        df_temp = ler_csv_seguro(caminho_csv)

        if df_temp is None:
            print(f"Não foi possível ler: {caminho_csv}")
            continue

        df_temp["ano_pasta"] = int(ano)
        df_temp["arquivo_origem"] = arquivo

        dfs.append(df_temp)

if not dfs:
    raise Exception("Nenhum CSV foi lido. Verifique a pasta data.")

df = pd.concat(dfs, ignore_index=True)

print(f"\nTotal de registros unificados: {len(df):,}")
print(f"Total de colunas unificadas: {len(df.columns)}")

if TARGET not in df.columns:
    raise Exception(f"Target '{TARGET}' não encontrada na base.")

# =========================================================
# 2. CONVERSÃO DAS COLUNAS NUMÉRICAS
# =========================================================

colunas_numericas = [
    "ano",
    "anoLicitacao",
    "anoProcesso",
    "ano_pasta",
    "quantidade",
    "valor",
    "valorEstimado",
    "valorHomologado",
    "valorTotal",
    "valorTotalReferencia",
    "valorTotalVencedor",
    "valorUnitario",
    "valorUnitarioReferencia",
    "valorUnitarioVencedor",
    "fator_ipca",
    "ipca_percentual",
]

for coluna in colunas_numericas:
    if coluna in df.columns:
        df[coluna] = df[coluna].apply(converter_numero_br)

# remove registros sem target
df = df.dropna(subset=[TARGET])

print(f"Registros com target válida: {len(df):,}")

if len(df) < 20000:
    print("ATENÇÃO: a base final ficou com menos de 20 mil registros.")

# =========================================================
# 3. CRIAÇÃO DE VARIÁVEIS DERIVADAS
# =========================================================

print("\nCriando variáveis derivadas...")

if "valorUnitarioReferencia" in df.columns:
    df["percentual_desconto_vencedor"] = (
        (
            df["valorUnitarioReferencia"] -
            df["valorUnitarioVencedor"]
        )
        / df["valorUnitarioReferencia"]
    ) * 100

    df["razao_valor_vencedor_referencia"] = (
        df["valorUnitarioVencedor"] /
        df["valorUnitarioReferencia"]
    )

if "fator_ipca" in df.columns:
    df["valor_unitario_corrigido_ipca"] = (
        df["valorUnitarioVencedor"] /
        df["fator_ipca"]
    )

if "ipca_percentual" in df.columns:
    df["valor_estimado_pela_inflacao"] = (
        df["valorUnitarioVencedor"] /
        (1 + (df["ipca_percentual"] / 100))
    )

if "quantidade" in df.columns:
    df["quantidade_em_lote"] = np.log(df["quantidade"] + 1)

if "valorTotalVencedor" in df.columns and "quantidade" in df.columns:
    df["valor_unitario_calculado"] = (
        df["valorTotalVencedor"] /
        df["quantidade"]
    )

if "valorTotalReferencia" in df.columns and "valorTotalVencedor" in df.columns:
    df["diferenca_total_referencia_vencedor"] = (
        df["valorTotalReferencia"] -
        df["valorTotalVencedor"]
    )

if "dataHomologacao" in df.columns and "dataPublicacao" in df.columns:
    df["dataHomologacao"] = pd.to_datetime(df["dataHomologacao"], errors="coerce")
    df["dataPublicacao"] = pd.to_datetime(df["dataPublicacao"], errors="coerce")

    df["duracao_processo_dias"] = (
        df["dataHomologacao"] - df["dataPublicacao"]
    ).dt.days

if "dataAberturaEnvelopes" in df.columns and "inicioRecebimentoEnvelopes" in df.columns:
    df["dataAberturaEnvelopes"] = pd.to_datetime(df["dataAberturaEnvelopes"], errors="coerce")
    df["inicioRecebimentoEnvelopes"] = pd.to_datetime(df["inicioRecebimentoEnvelopes"], errors="coerce")

    df["dias_recebimento_propostas"] = (
        df["dataAberturaEnvelopes"] - df["inicioRecebimentoEnvelopes"]
    ).dt.days

if "objeto" in df.columns:
    df["categoria_objeto"] = df["objeto"].apply(categorizar_objeto)

if "situacao" in df.columns:
    df["processo_homologado"] = (
        df["situacao"]
        .astype(str)
        .str.upper()
        .eq("HOMOLOGADO")
        .astype(int)
    )

if "registroPrecos" in df.columns:
    df["possui_registro_preco"] = (
        df["registroPrecos"]
        .astype(str)
        .str.upper()
        .eq("SIM")
        .astype(int)
    )

if "contratos" in df.columns:
    df["quantidade_contratos"] = df["contratos"].apply(contar_lista_json)
    df["possui_contrato"] = (df["quantidade_contratos"] > 0).astype(int)

if "participantes" in df.columns:
    df["quantidade_participantes"] = df["participantes"].apply(contar_lista_json)

if "documentosRelacionados" in df.columns:
    df["quantidade_documentos"] = df["documentosRelacionados"].apply(contar_lista_json)

df["ano_pandemia"] = df["ano_pasta"].isin([2020, 2021]).astype(int)

print("Variáveis derivadas criadas.")

# =========================================================
# 4. SELEÇÃO DE VARIÁVEIS PARA ANÁLISE
# =========================================================

colunas_originais_interesse = [
    TARGET,
    "ano_pasta",
    "anoLicitacao",
    "anoProcesso",
    "quantidade",
    "valorEstimado",
    "valorHomologado",
    "valorTotalReferencia",
    "valorTotalVencedor",
    "valorUnitarioReferencia",
    "fator_ipca",
    "ipca_percentual",
    "modalidade",
    "formaContratacao",
    "formaJulgamento",
    "tipoObjeto",
    "registroPrecos",
    "unidadeMedida",
    "situacao",
    "meioDivulgacao",
    "nomeEntidade",
    "fundamentoLegal",
    "categoria_objeto",
]

colunas_derivadas_interesse = [
    "percentual_desconto_vencedor",
    "razao_valor_vencedor_referencia",
    "valor_unitario_corrigido_ipca",
    "valor_estimado_pela_inflacao",
    "quantidade_em_lote",
    "valor_unitario_calculado",
    "diferenca_total_referencia_vencedor",
    "duracao_processo_dias",
    "dias_recebimento_propostas",
    "processo_homologado",
    "possui_registro_preco",
    "quantidade_contratos",
    "possui_contrato",
    "quantidade_participantes",
    "quantidade_documentos",
    "ano_pandemia",
]

colunas_usadas = [
    col for col in colunas_originais_interesse + colunas_derivadas_interesse
    if col in df.columns
]

df_modelo = df[colunas_usadas].copy()

print(f"\nColunas selecionadas antes do One Hot: {len(df_modelo.columns)}")

# =========================================================
# 5. ONE HOT ENCODING CONTROLADO
# =========================================================

categoricas = [
    "modalidade",
    "formaContratacao",
    "formaJulgamento",
    "tipoObjeto",
    "registroPrecos",
    "unidadeMedida",
    "situacao",
    "meioDivulgacao",
    "nomeEntidade",
    "fundamentoLegal",
    "categoria_objeto",
]

categoricas_existentes = [
    col for col in categoricas
    if col in df_modelo.columns
]

print("\nCategóricas usadas:")
print(categoricas_existentes)

# Agrupa categorias raras como OUTROS para não explodir a base
for col in categoricas_existentes:
    frequencias = df_modelo[col].value_counts(dropna=True)
    categorias_validas = frequencias[frequencias >= 100].index

    df_modelo[col] = df_modelo[col].where(
        df_modelo[col].isin(categorias_validas),
        "OUTROS"
    )

df_modelo = pd.get_dummies(
    df_modelo,
    columns=categoricas_existentes,
    drop_first=True,
    dtype=int
)

print(f"Colunas depois do One Hot: {len(df_modelo.columns)}")

# =========================================================
# 6. LIMPEZA FINAL
# =========================================================

df_modelo = df_modelo.replace([np.inf, -np.inf], np.nan)

df_numerico = df_modelo.select_dtypes(include=[np.number])

print(f"Total de variáveis numéricas na base final: {len(df_numerico.columns)}")

if TARGET not in df_numerico.columns:
    raise Exception(f"A target '{TARGET}' não ficou numérica.")

# =========================================================
# 7. CORRELAÇÃO
# =========================================================

corr_bruta = df_numerico.corr(numeric_only=True)[TARGET].drop(TARGET)

correlacoes = (
    corr_bruta
    .dropna()
    .sort_values(key=abs, ascending=False)
)

fortes = correlacoes[
    (correlacoes >= 0.3) |
    (correlacoes <= -0.3)
]

# =========================================================
# 8. OUTPUT
# =========================================================

print("\n" + "=" * 80)
print(f"TARGET: {TARGET}")
print("=" * 80)

print(f"\nRegistros finais analisados: {len(df_modelo):,}")
print(f"Total de variáveis numéricas analisadas: {len(correlacoes)}")
print(f"Variáveis com |correlação| >= 0.3: {len(fortes)}")

print("\nTOP 30 correlações:")
print(correlacoes.head(30))

print("\nVARIÁVEIS FORTEMENTE CORRELACIONADAS:")
if len(fortes) == 0:
    print("Nenhuma variável atingiu |corr| >= 0.3.")
else:
    for nome, valor in fortes.items():
        print(f"{nome:<70} {valor:.4f}")

# =========================================================
# 9. SALVAR ARQUIVOS
# =========================================================

df_modelo.to_csv(ARQUIVO_BASE_FINAL, index=False, encoding="utf-8-sig")
correlacoes.to_csv(ARQUIVO_CORRELACOES, header=["correlacao"], encoding="utf-8-sig")
fortes.to_csv(ARQUIVO_CORRELACOES_FORTES, header=["correlacao"], encoding="utf-8-sig")

print("\nArquivos gerados:")
print(f"- {ARQUIVO_BASE_FINAL}")
print(f"- {ARQUIVO_CORRELACOES}")
print(f"- {ARQUIVO_CORRELACOES_FORTES}")

# =========================================================
# 10. CHECK DOS REQUISITOS
# =========================================================

print("\n" + "=" * 80)
print("CHECK DOS REQUISITOS")
print("=" * 80)

print(f"Base final >= 20 mil registros? {'SIM' if len(df_modelo) >= 20000 else 'NÃO'}")
print(f"Pelo menos 25 variáveis analisadas? {'SIM' if len(correlacoes) >= 25 else 'NÃO'}")
print(f"Pelo menos 15 variáveis com |corr| >= 0.3? {'SIM' if len(fortes) >= 15 else 'NÃO'}")