import os
import pandas as pd

PASTA_DATA = "data"
ARQUIVO_SAIDA = "relatorio_colunas_por_ano.txt"

colunas_por_ano = {}

for ano in sorted(os.listdir(PASTA_DATA)):
    caminho_ano = os.path.join(PASTA_DATA, ano)

    if not os.path.isdir(caminho_ano):
        continue

    colunas_do_ano = set()

    for arquivo in os.listdir(caminho_ano):
        if not arquivo.lower().endswith(".csv"):
            continue

        caminho_csv = os.path.join(caminho_ano, arquivo)

        try:
            df = pd.read_csv(caminho_csv, nrows=0, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(caminho_csv, nrows=0, encoding="latin1")
        except Exception as e:
            print(f"Erro ao ler {caminho_csv}: {e}")
            continue

        for coluna in df.columns:
            colunas_do_ano.add(coluna.strip())

    colunas_por_ano[ano] = colunas_do_ano


todos_os_anos = sorted(colunas_por_ano.keys())
todas_colunas = set().union(*colunas_por_ano.values())


with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO DE COLUNAS POR ANO\n")
    f.write("=" * 80 + "\n\n")

    ano_anterior = None

    for ano in todos_os_anos:
        colunas_atuais = colunas_por_ano[ano]

        f.write(f"ANO: {ano}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total de colunas únicas no ano: {len(colunas_atuais)}\n\n")

        if ano_anterior is None:
            f.write("Primeiro ano analisado. Colunas encontradas:\n")
            for coluna in sorted(colunas_atuais):
                f.write(f"  - {coluna}\n")
        else:
            colunas_anteriores = colunas_por_ano[ano_anterior]

            novas = colunas_atuais - colunas_anteriores
            sumiram = colunas_anteriores - colunas_atuais
            mantidas = colunas_atuais & colunas_anteriores

            f.write(f"Comparação com {ano_anterior}:\n")
            f.write(f"Colunas mantidas: {len(mantidas)}\n")
            f.write(f"Colunas novas: {len(novas)}\n")
            f.write(f"Colunas que sumiram: {len(sumiram)}\n\n")

            f.write("Colunas novas neste ano:\n")
            if novas:
                for coluna in sorted(novas):
                    f.write(f"  + {coluna}\n")
            else:
                f.write("  Nenhuma coluna nova.\n")

            f.write("\nColunas que sumiram neste ano:\n")
            if sumiram:
                for coluna in sorted(sumiram):
                    f.write(f"  - {coluna}\n")
            else:
                f.write("  Nenhuma coluna sumiu.\n")

        f.write("\n" + "=" * 80 + "\n\n")
        ano_anterior = ano

    f.write("\nRESUMO GERAL\n")
    f.write("=" * 80 + "\n")
    f.write(f"Total de anos analisados: {len(todos_os_anos)}\n")
    f.write(f"Total de colunas diferentes encontradas na base inteira: {len(todas_colunas)}\n\n")

    f.write("Todas as colunas encontradas na base inteira:\n")
    for coluna in sorted(todas_colunas):
        f.write(f"  - {coluna}\n")

print(f"Relatório gerado com sucesso: {ARQUIVO_SAIDA}")