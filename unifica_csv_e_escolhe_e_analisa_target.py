import os
import pandas as pd
import numpy as np

PASTA_DATA = "data"
TARGET = "valorUnitarioVencedor"

dfs = []

def ler_csv_seguro(caminho_csv):
    encodings = ["utf-8", "latin1", "ISO-8859-1"]

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
    raise Exception("Nenhum CSV foi lido. Verifique se a pasta data está no lugar certo.")

df = pd.concat(dfs, ignore_index=True)

print(f"Total de registros unificados: {len(df):,}")
print(f"Total de colunas unificadas: {len(df.columns)}")

if len(df) < 20000:
    print("ATENÇÃO: a base final tem menos de 20 mil registros.")

if TARGET not in df.columns:
    raise Exception(f"Target '{TARGET}' não encontrada na base.")

def converter_numero_br(valor):
    if pd.isna(valor):
        return np.nan

    valor = str(valor).strip()

    if valor == "":
        return np.nan

    valor = valor.replace("R$", "").replace(" ", "")

    if "," in valor:
        valor = valor.replace(".", "").replace(",", ".")

    try:
        return float(valor)
    except:
        return np.nan


df[TARGET] = df[TARGET].apply(converter_numero_br)

df = df.dropna(subset=[TARGET])

print(f"Registros com target válida: {len(df):,}")

colunas_para_tentar = [
    "ano",
    "anoLicitacao",
    "anoProcesso",
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

for coluna in colunas_para_tentar:

    if coluna not in df.columns:
        continue

    # Só tenta converter se parecer numérica
    amostra = df[coluna].dropna().astype(str).head(20)

    possui_numero = amostra.str.contains(r"\d").mean()

    if possui_numero > 0.7:
        df[coluna] = df[coluna].apply(converter_numero_br)

CATEGORICAS_IMPORTANTES = [
    "modalidade",
    "tipoObjeto",
    "formaJulgamento",
    "formaContratacao",
    "situacao",
    "registroPrecos",
    "cidadeCertame",
    "estadoCertame",
]

categoricas_existentes = [
    col for col in CATEGORICAS_IMPORTANTES
    if col in df.columns
]

print("\nAplicando One Hot Encoding...")

df = pd.get_dummies(
    df,
    columns=categoricas_existentes,
    drop_first=True,
    dtype=int
)

print("\nCOLUNAS APÓS GET_DUMMIES:")
print(len(df.columns))

df_numerico = df.select_dtypes(include=[np.number])

print("\nTOTAL NUMÉRICAS:")
print(len(df_numerico.columns))

if TARGET not in df_numerico.columns:
    raise Exception(f"A target '{TARGET}' não ficou numérica.")

correlacoes = (
    df_numerico
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .dropna()
    .sort_values(key=abs, ascending=False)
)

fortes = correlacoes[(correlacoes >= 0.3) | (correlacoes <= -0.3)]


print("\n" + "=" * 80)
print(f"TARGET: {TARGET}")
print("=" * 80)

print(f"\nTotal de variáveis numéricas analisadas: {len(correlacoes)}")
print(f"Variáveis com correlação forte/moderada |corr| >= 0.3: {len(fortes)}\n")

for nome, valor in fortes.items():
    print(f"{nome:<50} {valor:.4f}")

correlacoes.to_csv("todas_correlacoes.csv", header=["correlacao"])
fortes.to_csv("correlacoes_fortes.csv", header=["correlacao"])

df.to_csv("base_unificada.csv", index=False, encoding="utf-8-sig")

print("\nArquivos gerados:")
print("- base_unificada.csv")
print("- todas_correlacoes.csv")
print("- correlacoes_fortes.csv")