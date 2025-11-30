import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


st.set_page_config(page_title="Music Cluster AI", layout="wide")

st.title("🎵 Agrupamento de Músicas com IA (Spotify Features)")
st.markdown("""
Este sistema utiliza **Machine Learning (K-Means)** para agrupar músicas com características sonoras semelhantes.
""")


st.sidebar.header("1. Dados e Parâmetros")

def gerar_dados_simulados():
    np.random.seed(42)
    n_samples = 200
    data = {
        'track_name': [f'Música Simulada {i}' for i in range(1, n_samples+1)],
        'danceability': np.random.rand(n_samples),
        'energy': np.random.rand(n_samples),
        'acousticness': np.random.rand(n_samples),
        'valence': np.random.rand(n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples)
    }
    df = pd.DataFrame(data)

    df['loudness'] = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
    return df

uploaded_file = st.sidebar.file_uploader("Upload do CSV (Spotify)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    if 'loudness' in df_raw.columns:
        min_l = df_raw['loudness'].min()
        max_l = df_raw['loudness'].max()
        df_raw['loudness'] = (df_raw['loudness'] - min_l) / (max_l - min_l)
    
    possible_name_cols = ['song']
    name_col = next((col for col in possible_name_cols if col in df_raw.columns), None)
    
    if name_col is None:
        df_raw['track_name'] = [f"Track {i}" for i in range(len(df_raw))]
        name_col = 'track_name'
    else:
        df_raw[name_col] = df_raw[name_col].astype(str).fillna("Desconhecida")

    target_features = ['danceability', 'energy', 'acousticness', 'valence', 'loudness']
    
    features_existentes = [col for col in target_features if col in df_raw.columns]
    
    if len(features_existentes) < 3:
        st.error("O CSV precisa ter colunas de audio features (danceability, energy, etc).")
        st.stop()
        
    df = df_raw.dropna(subset=features_existentes).copy()
    
else:
    st.sidebar.info("Utilizando dados simulados.")
    df = gerar_dados_simulados()
    features_existentes = ['danceability', 'energy', 'acousticness', 'valence', 'loudness']
    name_col = 'track_name'

n_clusters = st.sidebar.slider("Número de Clusters (K)", 2, 8, 4)

X = df[features_existentes]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
score = silhouette_score(X_scaled, clusters)

tab1, tab2, tab3 = st.tabs(["📊 Visualização", "🧠 Interpretação", "🔮 Predição"])

with tab1:
    st.header("Mapa de Músicas (PCA)")
    st.metric("Silhouette Score (Qualidade da Separação)", f"{score:.2f}")
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    df_viz = df.copy()
    df_viz['x_pca'] = pca_result[:, 0]
    df_viz['y_pca'] = pca_result[:, 1]
    df_viz['Cluster'] = df_viz['Cluster'].astype(str)

    fig_pca = px.scatter(
        df_viz, 
        x='x_pca', 
        y='y_pca', 
        color='Cluster',
        hover_name=name_col,
        hover_data=features_existentes, 
        title="Distribuição das Músicas",
        template="plotly_dark",
        height=600
    )
    fig_pca.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("Passe o mouse sobre os pontos para ver o nome da música e seus atributos.")

with tab2:
    st.header("Perfil dos Clusters")
    
    numeric_cols = features_existentes + ['Cluster']
    avg_df = df[numeric_cols].groupby('Cluster').mean().reset_index()
    
    col1, col2 = st.columns(2)
    
    for i in range(n_clusters):
        fig = go.Figure()
        vals = avg_df.loc[i, features_existentes].values.tolist()
        vals += vals[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=features_existentes + [features_existentes[0]],
            fill='toself', name=f'Cluster {i}'
        ))
        fig.update_layout(title=f"Cluster {i}", height=300, margin=dict(t=40, b=20, l=40, r=40))
        
        if i % 2 == 0: col1.plotly_chart(fig, use_container_width=True)
        else: col2.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Simulador")
    input_vals = []
    cols_input = st.columns(len(features_existentes))
    
    for i, col in enumerate(features_existentes):
        val = cols_input[i].slider(col, 0.0, 1.0, 0.5)
        input_vals.append(val)
        
    if st.button("Classificar"):
        entrada = np.array([input_vals])
        entrada_scaled = scaler.transform(entrada)
        pred = kmeans.predict(entrada_scaled)[0]
        st.success(f"Esta música seria do **Cluster {pred}**")