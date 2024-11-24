import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('social_media_1.csv')
for i in range(9):
    df = pd.concat([df, pd.read_csv(f'social_media_{i+2}.csv')])

desc = pd.read_csv('media_description.csv')

# Criando um pipeline com um transformer para as variáveis categóricas (OneHotEncoder)
# Usar StandardScaler para a variável contínua 'idade'

# Colunas categóricas
categorical_columns = ['categoria', 'genero', 'nacional']
interest_columns = ['Saúde e bem-estar',
       'Educação e aprendizado', 'Esportes', 'Fotografia', 'Fitness',
       'Carros e automóveis', 'Finanças e investimentos',
       'Atividades ao ar livre', 'Parentalidade e família', 'História',
       'Jogos', 'Música', 'Tecnologia', 'Moda', 'Faça você mesmo e artesanato',
       'Livros', 'Negócios e empreendedorismo', 'Natureza', 'Beleza',
       'Ciência', 'Alimentos e refeições', 'Causas sociais e ativismo',
       'Jardinagem', 'Filmes', 'Arte', 'Culinária', 'Viagem', 'Política',
       'Animais de estimação']  # Encontrar as colunas de interesse

# Definindo o transformer para as variáveis categóricas e as colunas de interesse
# Usaremos o OneHotEncoder para as variáveis categóricas e interesses
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),  # Categóricas (OneHotEncoding)
        ('interest', 'passthrough', interest_columns),  # Manter as colunas de interesse inalteradas
        ('age', StandardScaler(), ['idade'])  # Escalonamento da variável idade
    ])

# Preparando o modelo KNN dentro de um Pipeline
knn = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Pré-processamento
    ('classifier', KNeighborsClassifier(n_neighbors=11))  # Classificador KNN
])

# Separando as variáveis independentes (X) e a variável dependente (y)
X = df.drop(columns=['rede social', 'pais'])
y = df['rede social']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo KNN
knn.fit(X_train, y_train)

# Função para prever a rede social com base nos dados de entrada do usuário
def prever_rede_social(categoria_empresa, genero=None, localidade=None, faixa_etaria_min=None, faixa_etaria_max=None, interesses=[]):
    # Construindo o vetor de entrada para a previsão
    # Para cada variável, verificamos e montamos a entrada com base nos parâmetros fornecidos
    entrada_usuario = {
        'categoria': categoria_empresa,
        'genero': genero if genero else 'não especificado',  # Caso não tenha gênero, colocamos 'não especificado'
        'nacional': localidade if localidade else 'não especificado',  # Caso não tenha localidade, colocamos 'não especificado'
        'idade': faixa_etaria_min if faixa_etaria_min else 0,  # Caso não tenha idade mínima, assumimos 0
    }

    # Criamos um dicionário com interesses definidos como 1 e os outros como 0
    for col in interest_columns:
        if col in interesses:
            entrada_usuario[col] = 1
        else:
            entrada_usuario[col] = 0

    # Convertendo o dicionário para um DataFrame para a previsão
    entrada_usuario_df = pd.DataFrame([entrada_usuario])

    # Fazendo a previsão com o modelo KNN
    probabilidade = knn.predict_proba(entrada_usuario_df)
    

    # Obtendo as 3 redes sociais com maior probabilidade de serem eficientes para a empresa
    indices_top_3 = probabilidade[0].argsort()[-3:][::-1]
    redes_sociais_top_3 = knn.classes_[indices_top_3]
    probabilidades_top_3 = probabilidade[0][indices_top_3]

    return list(zip(redes_sociais_top_3, probabilidades_top_3))

# Função para exibir as redes sociais com descrições
def exibir_resultados(resultados):
    for i, (rede, probabilidade) in enumerate(resultados):
        st.subheader(f"{i+1}. {desc[desc['rede'].lower() == rede]['rede']}")
        # Descrições de cada rede social e público
        st.write(f"**Descrição**: {desc[desc['rede'].lower() == rede]['desc']}.")
        st.write(f"**Público**: {desc[desc['rede'].lower() == rede]['publico']}.")
        st.write("---")

# Streamlit - Interface

st.title("Previsão de Rede Social Ideal para o Público")

# Input do nome da empresa
nome_empresa = st.text_input("Qual o nome da sua empresa?", "")

if nome_empresa:
    st.write(f"Bem-vindo(a) {nome_empresa}!")

    # Seleção de categoria
    categoria = st.selectbox("Escolha a categoria da sua empresa:", df['categoria'].unique())
    
    # Seleção de gênero
    genero = st.selectbox("Escolha o gênero do público alvo:", ['homem', 'mulher', 'não especificado'])
    
    # Seleção de localidade (nacional ou internacional)
    localidade = st.selectbox("Escolha a localidade do público alvo:", ['nacional', 'internacional', 'não especificado'])
    
    # Faixa etária
    faixa_etaria_min = st.number_input("Idade mínima do público:", min_value=0, max_value=100, value=18)
    faixa_etaria_max = st.number_input("Idade máxima do público:", min_value=0, max_value=100, value=70)

    # Seleção de interesses
    interesses = st.multiselect("Escolha até 5 interesses para o seu público:",
                                interest_columns, max_selections=5)

    if st.button("Gerar Resultado"):
        if not interesses:
            st.error("Por favor, escolha pelo menos 1 interesse.")
        else:
            resultados = prever_rede_social(categoria, genero, localidade, faixa_etaria_min, faixa_etaria_max, interesses)
            exibir_resultados(resultados)
