import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import nltk

# Baixe stopwords na primeira execução
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_pt = stopwords.words('portuguese')

st.set_page_config(page_title="MVP IA para Recrutamento Decision", layout="wide")
st.title("🤖 MVP Inteligência Artificial para Recrutamento (Decision)")

st.sidebar.header("1️⃣ Upload dos arquivos base")
df_applicants = st.sidebar.file_uploader("Applicants (Excel)", type=["xlsx"])
df_vagas = st.sidebar.file_uploader("Vagas (Excel)", type=["xlsx"])
df_prospects = st.sidebar.file_uploader("Prospects (Excel)", type=["xlsx"])
df_prospects_recusado = st.sidebar.file_uploader("Prospects Recusados (Excel)", type=["xlsx"])

if not all([df_applicants, df_vagas, df_prospects, df_prospects_recusado]):
    st.info("⏳ Faça upload de todos os arquivos para iniciar a análise.")
    st.stop()

# --- 1. Carregamento dos arquivos
df_applicants = pd.read_excel(df_applicants)
df_vagas = pd.read_excel(df_vagas)
df_prospects = pd.read_excel(df_prospects)
df_prospects_recusado = pd.read_excel(df_prospects_recusado)

# --- 2. Montagem da base única
df_prospects['match'] = 1
df_prospects_recusado['match'] = 0
df_match = pd.concat([df_prospects, df_prospects_recusado], ignore_index=True)
df_full = df_match.merge(df_applicants, on='id_candidato', how='left')
df_full = df_full.merge(df_vagas, on='id_vaga', how='left')

st.write("🔎 **Amostra dos dados integrados**")
st.dataframe(df_full.head())

# =============================
# 3. Análise Exploratória Interativa
# =============================

st.header("📊 Análise de Palavras-Chave e Features dos Candidatos")
sns.set(style="whitegrid")

colunas_textuais = [
    'area_atuacao',
    'conhecimentos_tecnicos',
    'certificacoes',
    'cursos',
    'nivel_profissional',
    'nivel_academico_x',
    'nivel_ingles_x',
    'nivel_espanhol_x',
    'instituicao_ensino_superior'
]

palavras_ignoradas = {'fluente', 'intermediario', 'intermediário', 'basico', 'básico', 'avançado', 'português'}

def extrair_palavras(coluna, ignorar=palavras_ignoradas):
    palavras = []
    for texto in coluna.dropna():
        tokens = re.split(r'[,\n/;|-]', str(texto).lower())
        for palavra in tokens:
            palavra = re.sub(r'[^\w\s]', '', palavra).strip()
            if (
                palavra and 
                len(palavra) > 2 and 
                palavra not in ignorar
            ):
                palavras.append(palavra)
    return palavras

aba = st.selectbox("Escolha o tipo de análise de palavras-chave", [
    "Geral", "Apenas Contratados", "Apenas Não Contratados", "Idiomas em Contratados", "Idiomas em Não Contratados"
])

if aba == "Geral":
    st.subheader("Palavras-chave mais frequentes (todas as situações)")
    coluna_analise = st.selectbox("Escolha a coluna para analisar", colunas_textuais)
    palavras = extrair_palavras(df_full[coluna_analise])
    contagem = Counter(palavras).most_common(15)
    df_plot = pd.DataFrame(contagem, columns=['Palavra-chave', 'Frequência'])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, y='Palavra-chave', x='Frequência', palette='viridis', ax=ax)
    plt.title(f"🔠 Palavras-chave mais frequentes na coluna '{coluna_analise}'")
    plt.xlabel("Frequência")
    plt.ylabel("Palavra-chave")
    st.pyplot(fig)

elif aba == "Apenas Contratados":
    st.subheader("Palavras-chave mais frequentes entre Contratados")
    coluna_analise = st.selectbox("Coluna", colunas_textuais, key="contratados")
    df_contratados = df_full[df_full['match'] == 1]
    palavras = extrair_palavras(df_contratados[coluna_analise])
    contagem = Counter(palavras).most_common(15)
    df_plot = pd.DataFrame(contagem, columns=['Palavra-chave', 'Frequência'])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, y='Palavra-chave', x='Frequência', palette='Greens_d', ax=ax)
    plt.title(f"✅ Palavras-chave mais frequentes entre contratados – {coluna_analise}")
    st.pyplot(fig)

elif aba == "Apenas Não Contratados":
    st.subheader("Palavras-chave mais frequentes entre NÃO Contratados")
    coluna_analise = st.selectbox("Coluna", colunas_textuais, key="nao_contratados")
    df_nao_contratados = df_full[df_full['match'] == 0]
    palavras = extrair_palavras(df_nao_contratados[coluna_analise])
    contagem = Counter(palavras).most_common(15)
    df_plot = pd.DataFrame(contagem, columns=['Palavra-chave', 'Frequência'])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, y='Palavra-chave', x='Frequência', palette='Reds_d', ax=ax)
    plt.title(f"❌ Palavras-chave mais frequentes entre não contratados – {coluna_analise}")
    st.pyplot(fig)

elif aba == "Idiomas em Contratados":
    st.subheader("Outros idiomas mais frequentes em Candidatos Contratados")
    df_contratados = df_full[df_full['match'] == 1]
    tokens = extrair_palavras(df_contratados['outro_idioma_x'], ignorar=palavras_ignoradas)
    contagem = Counter(tokens).most_common(15)
    df_plot = pd.DataFrame(contagem, columns=['Palavra-chave', 'Frequência'])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, y='Palavra-chave', x='Frequência', palette='Greens_d', ax=ax)
    plt.title("Outros idiomas mais frequentes em candidatos aceitos")
    st.pyplot(fig)

elif aba == "Idiomas em Não Contratados":
    st.subheader("Outros idiomas mais frequentes em Candidatos NÃO Contratados")
    df_nao_contratados = df_full[df_full['match'] == 0]
    tokens = extrair_palavras(df_nao_contratados['outro_idioma_x'], ignorar=palavras_ignoradas)
    contagem = Counter(tokens).most_common(15)
    df_plot = pd.DataFrame(contagem, columns=['Palavra-chave', 'Frequência'])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, y='Palavra-chave', x='Frequência', palette='rocket', ax=ax)
    plt.title("Outros idiomas mais frequentes em candidatos não aceitos")
    st.pyplot(fig)

# --- Gráficos numéricos e categóricos ---
st.subheader("🔧 Engenharia de Features (pré-processamento)")

df_feat = df_full.copy()

def extrair_valor_remuneracao(valor):
    if pd.isna(valor):
        return np.nan
    match = re.search(r'[\d\.]+', str(valor))
    if match:
        num = match.group().replace('.', '').replace(',', '')
        try:
            return float(num)
        except:
            return np.nan
    return np.nan

with st.spinner("Processando a coluna de remuneração..."):
    df_feat['remuneracao'] = df_feat['remuneracao'].apply(extrair_valor_remuneracao)
    df_feat['remuneracao'] = df_feat['remuneracao'].fillna(df_feat['remuneracao'].median())
st.write("Distribuição da Remuneração (após limpeza):")
st.write(df_feat['remuneracao'].describe())

def contar_certificacoes(x):
    return len([item for item in re.split(r'[;,/|]', str(x)) if item.strip()])

df_feat['num_certificacoes'] = df_feat['certificacoes'].fillna('').apply(contar_certificacoes)
df_feat['num_outras_certificacoes'] = df_feat['outras_certificacoes'].fillna('').apply(contar_certificacoes)
df_feat['num_cursos'] = df_feat['cursos'].fillna('').apply(contar_certificacoes)

with st.spinner("Contando certificações e cursos..."):
    df_feat['num_certificacoes'] = df_feat['certificacoes'].fillna('').apply(contar_certificacoes)
    df_feat['num_outras_certificacoes'] = df_feat['outras_certificacoes'].fillna('').apply(contar_certificacoes)
    df_feat['num_cursos'] = df_feat['cursos'].fillna('').apply(contar_certificacoes)

st.write("Exemplo de contagem de certificações/cursos:")
st.write(df_feat[['certificacoes', 'num_certificacoes', 'outras_certificacoes', 'num_outras_certificacoes', 'cursos', 'num_cursos']].head())

with st.spinner("Convertendo ano de conclusão para numérico..."):
    df_feat['ano_conclusao'] = pd.to_numeric(df_feat['ano_conclusao'], errors='coerce').fillna(df_feat['ano_conclusao'].median())

st.write("Ano de conclusão (exemplo):")
st.write(df_feat['ano_conclusao'].describe())

with st.spinner("Calculando match de senioridade..."):
    df_feat['match_senioridade'] = df_feat.apply(
        lambda row: int(str(row['nivel_profissional']).lower() in str(row['titulo_vaga_x']).lower()),
        axis=1
    )

def contar_tecnologias_comuns(row):
    cand = set([c.strip() for c in re.split(r'[;/,\n|.]', str(row['conhecimentos_tecnicos']).lower()) if c.strip()])
    vaga = set([v.strip() for v in re.split(r'[;/,\n|.]', str(row.get('competencia_tecnicas_e_comportamentais', '')).lower()) if v.strip()])
    return len(cand & vaga)

with st.spinner("Calculando match de tecnologias..."):
    df_feat['match_tecnologias'] = df_feat.apply(contar_tecnologias_comuns, axis=1)

st.write("Exemplo de engenharia de features:")
st.write(df_feat[['remuneracao', 'num_certificacoes', 'num_outras_certificacoes', 'num_cursos', 'ano_conclusao', 'match_senioridade', 'match_tecnologias']].head())


st.header("📈 Análise Numérica e Categórica")
features_numericas = ['remuneracao', 'num_certificacoes', 'num_outras_certificacoes', 'num_cursos']
features_categoricas = ['nivel_profissional', 'nivel_academico_x', 'nivel_ingles_x']

st.subheader("Médias por situação de contratação")
st.write(df_feat.groupby('match')[features_numericas].mean())

st.subheader("Proporção de contratação por nível profissional")
st.write(df_feat.groupby('nivel_profissional')['match'].mean().sort_values(ascending=False))

for feat in features_categoricas:
    proporcao = df_feat.groupby(feat)['match'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=proporcao.index.astype(str), y=proporcao.values, palette="viridis", ax=ax)
    plt.title(f'Proporção de contratação por {feat}')
    plt.ylabel('Proporção de Contratados')
    plt.xlabel(feat)
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# --- Features avançadas (universidades, linguagens, etc) ---
st.header("🏫 Features Avançadas: Universidades e Linguagens de Programação")
top_universidades = [
    "USP", "UNICAMP", "UFRJ", "UFMG", "UNESP", "UFSC", "UFPR", "UFRGS", "UFPE", "UFC",
    "UFRN", "UNIFESP", "UFF", "UFBA", "UFSM", "UFES", "UFV", "UFPA", "UFABC", "UFPI",
    "UNB", "UFSCar", "UFAL", "UFOP", "PUC-Rio", "PUC-SP", "PUC-RS", "PUC-MG", "PUC-PR", "PUC-Campinas",
    "UFMA", "UFMT", "UFPB", "UFG", "UFS", "UFTM", "UFU", "UFPEL", "UENF", "UEPG",
    "UERJ", "UEMA", "UNIFAP", "UNIFAL", "UTFPR", "UNIOESTE", "UEM", "UNESP", "UNITAU", "UNIRIO"
]
top_universidades = [u.upper() for u in top_universidades]

df_feat['instituicao_ensino_superior'] = df_feat['instituicao_ensino_superior'].fillna('').astype(str)
df_feat['top_universidade'] = df_feat['instituicao_ensino_superior'].apply(
    lambda x: int(any(univ in x.upper() for univ in top_universidades))
)
st.write("Proporção de contratação para Universidades Prestigiadas")
st.write(df_feat.groupby('top_universidade')['match'].mean())

top_linguagens = [
    "python", "java", "javascript", "typescript", "c#", "c++", "c", "php", "ruby", "swift",
    "go", "kotlin", "r", "scala", "dart", "rust", "perl", "objective-c", "matlab", "shell"
]
df_feat['conhecimentos_tecnicos'] = df_feat['conhecimentos_tecnicos'].fillna('').astype(str)
for lang in top_linguagens:
    col_name = f'conhece_{lang.replace("#", "sharp").replace("++", "pp").replace("-", "_").replace(" ", "_")}'
    if col_name not in df_feat.columns:
        df_feat[col_name] = df_feat['conhecimentos_tecnicos'].str.lower().apply(lambda x: int(lang in x))

df_feat['num_top_linguagens'] = df_feat[
    [f'conhece_{lang.replace("#", "sharp").replace("++", "pp").replace("-", "_").replace(" ", "_")}' for lang in top_linguagens]
].sum(axis=1)

# Gera a coluna 'possui_experiencia'
if 'conhecimentos_tecnicos' in df_feat.columns:
    df_feat['possui_experiencia'] = df_feat['conhecimentos_tecnicos'].str.contains('experiência', case=False, na=False).astype(int)
else:
    df_feat['possui_experiencia'] = 0  # ou np.nan

# Gera a coluna 'num_palavras_curriculo'
if 'curriculo_pt' in df_feat.columns:
    df_feat['num_palavras_curriculo'] = df_feat['curriculo_pt'].fillna('').apply(lambda x: len(str(x).split()))
else:
    df_feat['num_palavras_curriculo'] = 0  # ou np.nan
    
st.write("Proporção de contratação por quantidade de linguagens mais utilizadas no currículo:")
st.write(df_feat.groupby('num_top_linguagens')['match'].mean())

# =============================
# 4. Pipeline do Modelo (IA)
# =============================

# Engenharia final de features para o modelo
# Features do modelo
features = [
    'remuneracao',
    'num_certificacoes',
    'num_outras_certificacoes',
    'num_cursos',
    'ano_conclusao',
    'match_tecnologias',
    'match_senioridade',
    'area_atuacao',
    'nivel_profissional',
    'nivel_academico_x',
    'nivel_ingles_x',
    'top_universidade',
    'conhece_python',
    'conhece_java',
    'conhece_javascript',
    'conhece_typescript',
    'conhece_csharp',
    'conhece_cpp',
    'conhece_c',
    'conhece_php',
    'conhece_ruby',
    'conhece_swift',
    'conhece_go',
    'conhece_kotlin',
    'conhece_r',
    'conhece_scala',
    'conhece_dart',
    'conhece_rust',
    'conhece_perl',
    'conhece_objective_c',
    'conhece_matlab',
    'conhece_shell',
    'num_top_linguagens',
    'possui_experiencia',
 'num_palavras_curriculo'
]
target = 'match'

for col in features:
    df_feat[col] = df_feat[col].fillna('desconhecido').astype(str)
    le = LabelEncoder()
    df_feat[col] = le.fit_transform(df_feat[col])

tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords_pt)
tfidf_matrix = tfidf.fit_transform(df_feat['curriculo_pt'].fillna(''))
X_base = df_feat[features].values
X_base = StandardScaler().fit_transform(X_base)
if not isinstance(X_base, csr_matrix):
    X_base = csr_matrix(X_base)
X_final = hstack([X_base, tfidf_matrix])
y = df_feat[target]

# SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_res, y_res = smote.fit_resample(X_final, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# VotingClassifier
voting = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, solver='saga', random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
], voting='soft')

with st.spinner("Treinando IA para recrutamento..."):
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    y_pred_proba = voting.predict_proba(X_test)[:, 1]

st.success("Modelo treinado! Veja as métricas:")
st.write("### Métricas do Modelo")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T
st.dataframe(df_report.style.format("{:.2f}"))
st.write("#### Matriz de Confusão")
st.write(confusion_matrix(y_test, y_pred))

st.header("🏆 Top 1% dos Candidatos Sugeridos pela IA")
top_percent = 0.01
top_n = max(1, int(len(y_pred_proba) * top_percent))
idx_top = np.argsort(y_pred_proba)[-top_n:][::-1]
df_top = pd.DataFrame(X_test[idx_top].todense() if hasattr(X_test, "todense") else X_test[idx_top])
df_top['score_contratado'] = y_pred_proba[idx_top]
st.dataframe(df_top)

csv = df_top.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Baixar Top 1% dos Candidatos (CSV)",
    data=csv,
    file_name='top_1porcento_candidatos.csv',
    mime='text/csv',
)

st.success("Deploy pronto!")

st.write("---")
st.write("© 2024 - MVP Decision - IA Recrutamento")
