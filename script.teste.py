import streamlit as st
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError as e:
    st.error("O módulo necessário não está instalado: " + str(e))
    st.stop()

# Leitura dos arquivos CSV
try:
    df = pd.read_csv('social_media_1.csv')
    for i in range(9):
        df = pd.concat([df, pd.read_csv(f'social_media_{i+2}.csv')])

    desc = pd.read_csv('media_description.csv')
except FileNotFoundError as e:
    st.error("Erro ao carregar os arquivos: " + str(e))
    st.stop()

# Definições para o pipeline
categorical_columns = ['categoria', 'genero', 'nacional']
interest_columns = ['Saúde e bem-estar', 'Educação e aprendizado', 'Esportes', 'Fotografia', 'Fitness',
                    'Carros e automóveis', 'Finanças e investimentos', 'Atividades ao ar livre', 'Parentalidade e família',
                    'História', 'Jogos', 'Música', 'Tecnologia', 'Moda', 'Faça você mesmo e artesanato', 'Livros',
                    'Negócios e empreendedorismo', 'Natureza', 'Beleza', 'Ciência', 'Alimentos e refeições',
                    'Causas sociais e ativismo', 'Jardinagem', 'Filmes', 'Arte', 'Culinária', 'Viagem', 'Política',
                    'Animais de estimação']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('interest', 'passthrough', interest_columns),
        ('age', StandardScaler(), ['idade'])
    ])

knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=11))
])

X = df.drop(columns=['rede social', 'pais'])
y = df['rede social']

# Divisão treino-teste e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn.fit(X_train, y_train)

# Função para prever a rede social
def prever_rede_social(categoria_empresa, genero=None, localidade=None, faixa_etaria_min=None, faixa_etaria_max=None, interesses=[]):
    entrada_usuario = {'categoria': categoria_empresa,
                       'genero': genero or 'indefinido',
                       'nacional': localidade or 'indefinido',
                       'idade': faixa_etaria_min or 0}
    for col in interest_columns:
        entrada_usuario[col] = 1 if col in interesses else 0
    entrada_usuario_df = pd.DataFrame([entrada_usuario])
    probabilidade = knn.predict_proba(entrada_usuario_df)
    indices_top_3 = probabilidade[0].argsort()[-3:][::-1]
    redes_sociais_top_3 = knn.classes_[indices_top_3]
    probabilidades_top_3 = probabilidade[0][indices_top_3]
    return list(zip(redes_sociais_top_3, probabilidades_top_3))

# Função para exibir os resultados
def exibir_resultados(resultados):
    for i, (rede, probabilidade) in enumerate(resultados):
        st.subheader(f"{i+1}. {rede}")
        descricao = desc[desc['rede'].str.lower() == rede.lower()]['descricao'].values
        publico = desc[desc['rede'].str.lower() == rede.lower()]['publico'].values
        st.write(f"**Descrição**: {descricao[0] if descricao else 'Não disponível'}.")
        st.write(f"**Público**: {publico[0] if publico else 'Não disponível'}.")
        st.write("---")

# Interface no Streamlit
st.title("Previsão de Rede Social Ideal para o Público")
nome_empresa = st.text_input("Qual o nome da sua empresa?", "")
if nome_empresa:
    st.write(f"Bem-vindo(a) {nome_empresa}!")
    categoria = st.selectbox("Escolha a categoria da sua empresa:", df['categoria'].unique())
    genero = st.selectbox("Escolha o gênero do público alvo:", ['homem', 'mulher', 'indefinido'])
    localidade = st.selectbox("Escolha a localidade do público alvo:", ['nacional', 'internacional', 'indefinido'])
    faixa_etaria_min = st.number_input("Idade mínima do público:", min_value=0, max_value=100, value=18)
    faixa_etaria_max = st.number_input("Idade máxima do público:", min_value=0, max_value=100, value=70)
    interesses = st.multiselect("Escolha até 5 interesses para o seu público:", interest_columns, max_selections=5)
    if st.button("Gerar Resultado"):
        if not interesses:
            st.error("Por favor, escolha pelo menos 1 interesse.")
        else:
            resultados = prever_rede_social(categoria, genero, localidade, faixa_etaria_min, faixa_etaria_max, interesses)
            exibir_resultados(resultados)
