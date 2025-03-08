import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from transformers import pipeline
from bertopic import BERTopic
from openai import OpenAI
import nltk
from collections import Counter
import io
from dotenv import load_dotenv
import os

# Configuración de la página
st.set_page_config(page_title="Streamlit App", layout="wide")

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de SpaCy para español
nlp = spacy.load("es_core_news_sm")

# Función para leer el archivo Excel
def read_excel(file):
    return pd.read_excel(file)

# Función para mostrar un muestreo de los datos
def show_sample(df):
    st.write("Primeras 20 filas del dataset:")
    st.write(df.head(20))

# Función para mostrar los nombres de las columnas y sus tipos de datos
def show_columns(df):
    st.write("Nombres de las columnas y sus tipos de datos:")
    column_info = df.dtypes.apply(lambda x: 'Numérico' if pd.api.types.is_numeric_dtype(x) else 'Categórico')
    st.write(column_info)

# Función para cambiar los nombres de las columnas
def change_column_names(df):
    st.write("Cambiar nombres de las columnas:")
    new_column_names = {}
    for col in df.columns:
        new_name = st.text_input(f"Nuevo nombre para '{col}':", col)
        new_column_names[col] = new_name
    if st.button("Aplicar cambios"):
        df.columns = [new_column_names[col] for col in df.columns]
        st.success("Nombres de columnas actualizados.")
    return df

# Función para generar gráficos básicos de las columnas numéricas
def plot_data(df):
    st.write("Gráficos básicos de las columnas numéricas:")
    numeric_columns = df.select_dtypes(include='number').columns
    for col in numeric_columns:
        st.write(f"Gráfico de '{col}':")
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True)
        st.pyplot(plt)

# Función para mostrar un mapa de calor de correlaciones
def show_correlations(df):
    st.write("Mapa de calor de correlaciones:")
    numeric_columns = df.select_dtypes(include='number').columns
    correlations = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

# Función para mostrar gráficos cruzados de las variables más correlacionadas
def show_cross_plots(df):
    st.write("Gráficos cruzados de las variables más correlacionadas:")
    numeric_columns = df.select_dtypes(include='number').columns
    correlations = df[numeric_columns].corr()
    high_corr = correlations.unstack().sort_values(ascending=False).drop_duplicates()
    high_corr = high_corr[(high_corr > 0.5) & (high_corr < 1)]

    for pair in high_corr.index:
        col1, col2 = pair
        st.write(f"Gráfico cruzado de '{col1}' vs '{col2}':")
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df[col1], y=df[col2])
        st.pyplot(plt)

# Función para generar gráficos personalizados
def custom_plots(df):
    st.write("Gráficos personalizados:")
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Selecciona las columnas para graficar:", columns)
    plot_type = st.selectbox("Selecciona el tipo de gráfico:", ["Histograma", "Scatter Plot", "Box Plot"])

    if st.button("Generar gráfico"):
        for col in selected_columns:
            st.write(f"Gráfico de '{col}':")
            plt.figure(figsize=(10, 5))
            if plot_type == "Histograma":
                sns.histplot(df[col], kde=True)
            elif plot_type == "Scatter Plot":
                sns.scatterplot(x=df.index, y=df[col])
            elif plot_type == "Box Plot":
                sns.boxplot(x=df[col])
            st.pyplot(plt)

# Función para descargar los gráficos personalizados en un archivo HTML
def download_plots(df):
    st.write("Descargar gráficos personalizados en un archivo HTML:")
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Selecciona las columnas para graficar:", columns)
    plot_type = st.selectbox("Selecciona el tipo de gráfico:", ["Histograma", "Scatter Plot", "Box Plot"])

    if st.button("Generar y descargar gráficos"):
        html_content = "<html><body>"
        for col in selected_columns:
            plt.figure(figsize=(10, 5))
            if plot_type == "Histograma":
                sns.histplot(df[col], kde=True)
            elif plot_type == "Scatter Plot":
                sns.scatterplot(x=df.index, y=df[col])
            elif plot_type == "Box Plot":
                sns.boxplot(x=df[col])
            plt.title(f"Gráfico de '{col}'")
            plt.savefig(f"{col}.png")
            html_content += f"<img src='{col}.png'><br>"
        html_content += "</body></html>"

        st.download_button(
            label="Descargar gráficos como HTML",
            data=html_content,
            file_name="graficos.html",
            mime="text/html"
        )

# Función para leer y analizar verbatims
def read_verbatims(df, verbatim_col):
    st.write("Análisis de verbatims:")

    # Eliminar tildes en todas las columnas de texto
    if st.button("Clean verbatims"):
        df = df.applymap(lambda x: unidecode(str(x)) if isinstance(x, str) else x)
        df[f'{verbatim_col}_clean'] = df[verbatim_col].copy()

        # Función de preprocesamiento con SpaCy
        def preprocess_text(text):
            if isinstance(text, str) and text.strip():
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
                tokens = word_tokenize(text)

                stop_words = set(stopwords.words('spanish'))

                # Aplicar lematización con SpaCy
                doc = nlp(' '.join(tokens))
                tokens = [token.lemma_ for token in doc if token.text not in stop_words]

                return ' '.join(tokens)
            return None

        # Aplicar preprocesamiento
        df[f'{verbatim_col}_clean'] = df[f'{verbatim_col}_clean'].apply(preprocess_text)
        st.success("Verbatims limpiados.")

    # Generar una nube de palabras
    if st.button("Wordcloud"):
        all_words = ' '.join(df[f'{verbatim_col}_clean'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nube de Palabras de los Verbatims")
        st.pyplot(plt)

    # Mostrar las 30 palabras más frecuentes
    if st.button("Keywords"):
        all_words = ' '.join(df[f'{verbatim_col}_clean'].dropna())
        word_counts = Counter(all_words.split())
        word_freq_df = pd.DataFrame(word_counts.items(), columns=["Palabra", "Frecuencia"]).sort_values(by="Frecuencia", ascending=False).head(30)

        plt.figure(figsize=(10, 5))
        sns.barplot(x="Frecuencia", y="Palabra", data=word_freq_df, palette="viridis")
        plt.title("Top 30 Palabras Más Frecuentes")
        st.pyplot(plt)

        st.write("\nFrecuencia de las 30 palabras más repetidas:")
        st.write(word_freq_df)

    # Modelo de topic modeling con BERTopic
    if st.button("Classify by topic"):
        filtered_df = df[df[f'{verbatim_col}_clean'].notna()].copy()
        model = BERTopic()
        topics, probs = model.fit_transform(filtered_df[f'{verbatim_col}_clean'])

        # Asignar los topics a cada verbatim
        filtered_df['topics'] = topics

        # Obtener los términos más representativos de cada tema
        topic_terms = model.get_topics()

        # Agrupar verbatims por topic y seleccionar hasta 20 ejemplos por topic
        grouped_verbatims = filtered_df.groupby('topics')[verbatim_col].apply(lambda x: x.sample(min(len(x), 20)).tolist()).to_dict()

       # Cargar variables de entorno desde el archivo .env
        load_dotenv()
       
       # Obtener la clave de API desde una variable de entorno
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Conectar con OpenAI para generar nombres de topics
        client = OpenAI(api_key=openai_api_key)

        def generate_topic_name(topic_id, verbatims):
            """
            Usa GPT para generar un nombre de topic basado en los verbatims del cluster y las palabras clave de BERTopic.
            """
            terms = [term for term, _ in topic_terms.get(topic_id, [])]
            prompt = f"""Estás analizando respuestas de una encuesta y dando un nombre al tema del que hablan estos verbatims.
            Aquí tienes 20 respuestas agrupadas bajo el mismo tema, junto con algunas palabras clave generadas por un modelo de topic modeling:

            - Palabras clave del modelo: {', '.join(terms[:5])}
            - Ejemplos de respuestas:
            {verbatims}

            Basándote en esto, genera un nombre breve, conciso y claro para el tema que represente bien su contenido.
            Solo responde con el nombre del tema, sin añadir nada más.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Eres un asistente experto en análisis de encuestas."},
                          {"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.5
            )

            return response.choices[0].message.content.strip()

        # Generar nombres de topics con GPT
        topic_names = {topic_id: generate_topic_name(topic_id, verbatims) for topic_id, verbatims in grouped_verbatims.items()}

        # Asignar los nombres generados al dataset
        filtered_df['topic_labels'] = filtered_df['topics'].map(topic_names)

        # Fusionar los topics de vuelta en el dataframe original
        df = df.merge(filtered_df[['topics', 'topic_labels']], how='left', left_index=True, right_index=True)

        # Visualizar la distribución de topics
        df_filtered = df[df['topics'].notna()]
        topic_counts = df_filtered['topic_labels'].value_counts().reset_index()
        topic_counts.columns = ['Topic Label', 'Count']

        st.write("Distribución de Temas Refinados:")
        st.bar_chart(topic_counts.set_index('Topic Label'))

        st.write("\nTopic Frequency Table (Refined with GPT):")
        st.write(topic_counts)

    # Modelo de análisis de sentimiento con BERT Multilingual
    if st.button("Classify by sentiment"):
        sentiment_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

        def analyze_sentiment(text):
            if isinstance(text, str) and text.strip():
                sentiment = sentiment_model(text[:512])[0]['label']
                return 'positive' if '5' in sentiment or '4' in sentiment else 'negative' if '1' in sentiment or '2' in sentiment else 'neutral'
            return 'neutral'

        df['sentiment'] = df[verbatim_col].apply(analyze_sentiment)

        # Visualizar la distribución de sentimientos
        df_filtered = df[df['topics'].notna()]
        sentiment_counts = df_filtered['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        st.write("Distribución de Sentimientos:")
        st.bar_chart(sentiment_counts.set_index('Sentiment'))

        st.write("\nSentiment Frequency Table:")
        st.write(sentiment_counts)

    return df

# Función para descargar el archivo Excel modificado
def download_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    st.download_button(
        label="Descargar archivo Excel modificado",
        data=output,
        file_name="datos_modificados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Barra lateral para la navegación
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona una página:", ["Inicio", "Análisis de Datos", "Gráficos", "Verbatims", "Descargas"])

# Cargar el archivo Excel
uploaded_file = st.sidebar.file_uploader("Sube un archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    df = read_excel(uploaded_file)

    if page == "Inicio":
        st.title("Inicio")
        st.write("Bienvenido a la aplicación de análisis de datos. Usa la barra lateral para navegar entre las diferentes secciones.")

    elif page == "Análisis de Datos":
        st.title("Análisis de Datos")
        if st.button("Sample"):
            show_sample(df)
        if st.button("Read columns"):
            show_columns(df)
        if st.button("Change names"):
            df = change_column_names(df)

    elif page == "Gráficos":
        st.title("Gráficos")
        if st.button("Plot"):
            plot_data(df)
        if st.button("Correlations"):
            show_correlations(df)
        if st.button("Cross plots"):
            show_cross_plots(df)
        if st.button("Custom plots"):
            custom_plots(df)

    elif page == "Verbatims":
        st.title("Análisis de Verbatims")
        verbatim_col = st.selectbox("Selecciona la columna con los verbatims:", df.columns)
        if verbatim_col:
            df = read_verbatims(df, verbatim_col)

    elif page == "Descargas":
        st.title("Descargas")
        if st.button("Download plots"):
            download_plots(df)
        if st.button("Download file"):
            download_excel(df)

else:
    st.warning("Por favor, sube un archivo Excel para comenzar.")
