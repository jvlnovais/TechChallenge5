MVP IA para Recrutamento — Decision
Projeto desenvolvido para otimizar o processo de recrutamento de talentos na área de TI utilizando Inteligência Artificial, em parceria com a empresa Decision.

🚀 Objetivo
O objetivo deste projeto é entregar uma solução de IA capaz de:

Classificar candidatos conforme o potencial de contratação

Analisar grandes volumes de currículos de forma automática e escalável

Identificar padrões de sucesso em contratações anteriores

Apoiar o time de RH nas tomadas de decisão com base em dados concretos


🛠️ Tecnologias e principais bibliotecas
Python 3.10+

Streamlit (deploy web)

Pandas, NumPy, Scikit-Learn, Imbalanced-Learn

NLTK (para análise de texto)

Matplotlib, Seaborn


⚙️ Como executar
Clone o repositório

bash
Copiar
Editar
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
Instale as dependências

bash
Copiar
Editar
pip install -r requirements.txt
Execute o app Streamlit

bash
Copiar
Editar
streamlit run app.py
Faça upload dos arquivos de dados

applicants.xlsx

vagas.xlsx

prospects.xlsx

prospects_recusado.xlsx

📄 Estrutura do Projeto
bash
Copiar
Editar
.
├── app.py               # Código principal Streamlit
├── requirements.txt     # Lista de dependências Python
├── README.md
├── data/                # (Opcional) Pasta para colocar os arquivos Excel


✨ Funcionalidades
Upload de arquivos de candidatos, vagas e resultados históricos

Processamento automático de currículos e extração de informações

Treinamento de modelo de Machine Learning com balanceamento

Visualização de métricas (acurácia, recall, f1-score, matriz de confusão)

Download do Top 1% dos candidatos recomendados pelo modelo

Interface web amigável via Streamlit


💡 Como funciona?
O usuário faz upload dos arquivos.

O sistema integra as bases, faz feature engineering e análise de texto dos currículos.

Um modelo de ensemble (VotingClassifier) é treinado com balanceamento (SMOTE).

O app mostra os principais indicadores do modelo e sugere automaticamente os melhores candidatos.


Observação: Este projeto foi desenvolvido para fins acadêmicos/profissionais, demonstrando como IA pode transformar o processo de recrutamento.


📫 Contato
Nome: Vinicius M. Novais

LinkedIn: [Seu LinkedIn](https://www.linkedin.com/in/vinicius-m-novais-0197b2295)

Email: jvlnovais123@gmail.com
