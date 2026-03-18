# 📊 Análise de Licitações Públicas com Streamlit

Este projeto realiza uma análise estatística de dados públicos de licitações e contratos, com foco em entender:

> **Os valores das licitações estão bem distribuídos ou concentrados em poucos processos/empresas?**

A aplicação foi desenvolvida utilizando **Python + Streamlit**, permitindo visualização interativa dos dados.

---

## 🚀 Como executar o projeto

### ✅ 1. Clonar o repositório

```bash
git clone https://github.com/Vinicius-Lummertz/TrabalhoAnaliseLicitacoes.git
cd SEU-REPO
🐍 2. Instalar Python (caso não tenha)

Baixe em: https://www.python.org/downloads/

Durante a instalação:

✔ Marque "Add Python to PATH"

📦 3. Verificar se o pip está instalado
pip --version

Se der erro:

python -m ensurepip --upgrade
🧪 4. Criar ambiente virtual (venv)
Criar:
python -m venv venv

- Ativar -
Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

📚 5. Instalar dependências

Se existir requirements.txt:

pip install -r requirements.txt

Ou manualmente:

pip install streamlit pandas matplotlib seaborn numpy

▶️ 6. Rodar a aplicação
streamlit run app.py

Acesse no navegador:

http://localhost:8501
📁 Estrutura do projeto
.
├── app.py
├── requirements.txt
├── README.md
├── 27_processoslicitatorios-34862.csv
├── 27_processoslicitatorios-34864.csv
├── 27_contratos-87437.csv
📊 Funcionalidades

📦 Unificação de múltiplas bases públicas

📉 Boxplot com análise de:

Quartis

Mediana

Outliers

📈 Distribuição dos valores

🏢 Ranking de empresas por valor contratado

🔥 Heatmap de correlação

🎯 Storytelling para apresentação

🧠 Bibliotecas/Tecnologias utilizadas

Python
Streamlit
Pandas
Matplotlib
Seaborn

⚠️ Problemas comuns
❌ Erro: missing ScriptRunContext

Você rodou errado.

Use:

streamlit run app.py
❌ pip não funciona

Tente:

python -m pip install --upgrade pip
❌ Erro com datas (timezone)

Já tratado no código com:

pd.to_datetime(..., utc=True).dt.tz_localize(None)
💡 Observação

Os dados utilizados são públicos e foram extraídos de portais de transparência.
