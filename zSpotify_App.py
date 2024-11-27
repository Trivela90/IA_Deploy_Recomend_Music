# ---- Biblliotecas ----
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import random
import streamlit as st

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import silhouette_score
from joblib import dump, load

# from yellowbrick.cluster import KElbowVisualizer # cluster visualizer (requer pip install setuptools)
# --------------------------------------------------------------------

# streamlit run c:/IC_Petrobras/CODE/Codigo1_begin/zSpotify_App.py

# Importando dados prontos
music_orig_noDuplicate = pd.read_csv("music_origNoDuplicate.csv")
music_test1 = pd.read_csv("music_test1.csv")

std_scaler = load('std_scaler.bin') # Normalização

hist_rock = load("hist_rock.bin")
hist_erudito = load("hist_erudito.bin")
hist_rap = load("hist_rap.bin")
std_scaler = load('std_scaler.bin')

features = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
            'loudness', 'speechiness', 'tempo', 'popularity']


# ----- Funcao que mostra recomendacoes caso o 
# ----- usuario possua historico de musicas no Spotify
def recomendar_para_usuario_com_historico(historico, df, n_recomendacoes, peso_popularidade, peso_instrum, peso_valence):
    try:
        # Filtra o DataFrame para encontrar músicas exatas com base no histórico fornecido
        historico_features = df[df['musica_artista_ano'].isin(historico)][features]
        
        # Verifica se o histórico contém dados válidos após a filtragem
        if historico_features.empty:
            return "Não foi possível encontrar músicas com base no histórico fornecido."
        
        # Calcula a média das características das músicas ouvidas
        media_features = historico_features.mean()
        
        # Ajusta o peso da popularidade, instrumentalidade e positividade
        media_features['popularity'] *= peso_popularidade
        media_features['instrumentalness'] *= peso_instrum
        media_features['valence'] *= peso_valence
        
        # Normaliza a média das características
        media_features_scaled = std_scaler.transform(pd.DataFrame(media_features).T)
        
        # Calcula similaridade entre a média ponderada e as músicas no dataset
        similaridades = cosine_similarity(media_features_scaled, std_scaler.transform(df[features]))[0]
        
        # Seleciona os índices das músicas com maior similaridade
        indices = similaridades.argsort()[-n_recomendacoes:][::-1]
        
        # Retorna as músicas recomendadas com base na similaridade e popularidade
        return df.iloc[indices][['musica_artista_ano', 'popularity']]
    
    except IndexError:
        return "Erro ao calcular as recomendações."
# ---------------------------------------------------------------



# ----- Funcao que mostra recomendacoes para o 
# ----- usuario que acabou de se cadastrar (sem historico de musicas)
def recomendar_para_novo_usuario(df, n_recomendacoes, amigos = None):
    if amigos:
        # Filtra o DataFrame para encontrar músicas que correspondam ao histórico dos amigos
        recomendacoes = df[df['musica_artista_ano'].isin(amigos)]
        
        # Verifica se o número de recomendações é suficiente
        if len(recomendacoes) < n_recomendacoes:
            # Completa com músicas populares, excluindo as já recomendadas
            populares = df[~df['musica_artista_ano'].isin(recomendacoes['musica_artista_ano'])]
            populares = populares.sort_values(by='popularity', ascending=False)
            recomendacoes = pd.concat([recomendacoes, populares.head(n_recomendacoes - len(recomendacoes))])

            # Seleciona as colunas principais e limita ao número de recomendações
            return recomendacoes[['musica_artista_ano', 'popularity']].head(n_recomendacoes)

        # Se ha mais recomendações que o a interface pode recomendar
        if len(recomendacoes) >= n_recomendacoes:
            # Escolha musicas aleatoriamente das recomendacoes de amigos
            recomendacoes_index = recomendacoes.index.tolist()
            sorteio = random.sample(recomendacoes_index, n_recomendacoes)
            return recomendacoes.loc[sorteio][['musica_artista_ano', 'popularity']]
            
    else:
        # Recomendação baseada apenas nas músicas mais populares para novos usuários sem amigos
        return df.sort_values(by='popularity', ascending=False).head(n_recomendacoes)[['musica_artista_ano', 'popularity']]
# ---------------------------------------------------------------



# # ----- Funcao recomendacao
# def recomendar_musicas(df, n_recomendacoes, peso_popularidade, peso_instrum, peso_valence, historico=None, amigos=None):
#     if historico:
#         print(f"Recomendações para usuário com histórico:")
#         return recomendar_para_usuario_com_historico(historico, df, n_recomendacoes, peso_popularidade, peso_instrum, peso_valence)
#     else:
#         print(f"Recomendações para novo usuário:")
#         return recomendar_para_novo_usuario(df, n_recomendacoes, amigos)
# # ---------------------------------------------------------------



# ----- Funcao que cria playlist de amigos
def sortear_musicas(df):
    tamanho_historico = random.randint(1,100)
    sorteio = df.sample(n = tamanho_historico)
    return sorteio['musica_artista_ano'].tolist()
# ---------------------------------------------------------------


# ---- Deploy ----
st.set_page_config(layout = "wide")             # Tamanho do layout (margens)
st.title("Recomendação de músicas do Spotify")  # Título

col1, col2 = st.columns(2, gap = "large")

#------------------------------------------------------
# --------------------- Coluna da esquerda (parâmetros)
with col1:
    tipo_conta = st.selectbox("A recomendação é para", ("Conta criada recentemente",
                                                        "Conta com registro de músicas escutadas"))


    popularidade = st.slider("Deseja músicas populares?", 0.5, 1.2, (1.0))
    instrumental = st.slider("Deseja músicas instrumentais?", 0.5, 1.2, (1.0))
    alegre = st.slider("Deseja músicas alegres?", 0.5, 1.2, (1.0))


    if tipo_conta == "Conta com registro de músicas escutadas":
        st.session_state.disabled = False
    else:
        st.session_state.disabled = True


    escolha_historico = st.selectbox("A pesquisa deve ser feita com base em que tipo de histórico?",
                                    ("Histório de rock", 
                                    "Histórico de músicas eruditas",
                                    "Histórico de rap"),
                                    disabled = st.session_state.disabled)

    # if st.session_state.disabled == True:
    #     hist_amigo = sortear_musicas(music_test1)
    # # st.write(escolha_historico)

    num_recom = st.number_input("Quantas recomendações você deseja?",
                                min_value = 5, max_value = 100)
    
    tem_amigos = st.checkbox("Possui amigos na conta",
                             disabled = not st.session_state.disabled)
    
    recomenda_button = st.button("Recomendar")


    



# ----- Funcao que molda o print das recomendações
def formatar_recomendacoes(resultados):
    """Formata e imprime as recomendações de forma legível."""
    if isinstance(resultados, str):  # Verifica se resultados é uma mensagem de erro
        return resultados
    
    recomendacoes_formatadas = []
    for _, row in resultados.iterrows():
        musica = row['musica_artista_ano']
        popularidade = row['popularity']
        recomendacoes_formatadas.append(f"Música: {musica} ---- [Popularidade: {popularidade}]\n")
    
    return "\n".join(recomendacoes_formatadas)
# ---------------------------------------------------------------



#------------------------------------------------------
# --------------------- Coluna da esquerda (parâmetros)
with col2:

    if recomenda_button:

        if escolha_historico == "Histório de rock":
            historico_user = hist_rock

        if escolha_historico == "Histórico de músicas eruditas":
            historico_user = hist_erudito

        if escolha_historico == "Histórico de rap":
            historico_user = hist_rap

    
        if tipo_conta == "Conta com registro de músicas escutadas":
            recomendacoes_usuario = recomendar_para_usuario_com_historico(historico_user, 
                                                                          music_orig_noDuplicate, 
                                                                          num_recom, 
                                                                          popularidade, 
                                                                          instrumental, 
                                                                          alegre)
            st.write("Recomendações para usuário com histórico:")
            st.write("=" * 85 + "\n")
            st.write(formatar_recomendacoes(recomendacoes_usuario))
            st.write("=" * 85 + "\n")
        else:
            if tem_amigos:
                hist_amigo = sortear_musicas(music_test1)

                recomendacoes_novo_usuario = recomendar_para_novo_usuario(df = music_orig_noDuplicate, 
                                                                          n_recomendacoes = num_recom, 
                                                                          amigos = hist_amigo)
                st.write("Recomendações para novo usuário com amigos:")
                st.write("=" * 85 + "\n")
                st.write(formatar_recomendacoes(recomendacoes_novo_usuario))
                st.write("=" * 85 + "\n")
            else:
                recomendacoes_novo_usuario_sem_amigo = recomendar_para_novo_usuario(music_orig_noDuplicate, 
                                                                                    num_recom)
                st.write("Recomendações para novo usuário sem amigos:")
                st.write("=" * 85 + "\n")
                st.write(formatar_recomendacoes(recomendacoes_novo_usuario_sem_amigo))
                st.write("=" * 85 + "\n")
