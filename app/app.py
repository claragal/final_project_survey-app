import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from transformers import pipeline
import nltk
from collections import Counter
from bertopic import BERTopic
from openai import OpenAI
from dotenv import load_dotenv
import os
import io  # Librería para manejar archivos en memoria

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de SpaCy para español
nlp = spacy.load("es_core_news_sm")

# Función para cargar el archivo
def upload_file():
    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=['xlsx'], key="uploader")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        return df
    return None

# Función para mostrar las primeras 20 filas
def show_sample(df):
    st.write("Echa un vistazo a las primeras 20 filas de tu dataset:")
    st.write(df.head(20))

# Función para mostrar las columnas y sus tipos
def show_columns(df):
    columns_info = pd.DataFrame({
        'Tipo': df.dtypes
    })
    st.write("Información de las columnas:")
    st.write(columns_info)

# Función para cambiar los nombres de las columnas
def change_column_names(df):
    st.write("¿No te gusta el nombre de los campos?")
    new_names = {}
    for col in df.columns:
        new_name = st.text_input(f"Cambiar nombre de '{col}' a:", col, key=f"rename_{col}")
        new_names[col] = new_name
    if st.button('Aplicar cambios', key="apply_changes"):
        df.rename(columns=new_names, inplace=True)
        st.session_state.df = df
        st.success("Nombres de columnas actualizados.")
    return df

# Función para graficar datos
def plot_data(df):
    st.write("Gráficos básicos de todas las columnas:")
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    df.columns = df.columns.str.replace(' ', '_')
    for col in df.columns:
        # Truncar el nombre de la variable si es demasiado largo
        short_name = col[:20] + "..." if len(col) > 60 else col

        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() == 1:
                st.warning(f"La columna '{col}' tiene un solo valor único y no se graficará.")
                continue
            # Gráfico de histograma para columnas numéricas
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(col, bin=alt.Bin(maxbins=20), title=short_name),  # Usar el nombre truncado
                y='count()',
                tooltip=[col, 'count()']
            ).properties(
                width=600,
                height=400,
                title=f"Distribución de {short_name}"  # Usar el nombre truncado en el título
            )
        else:
            # Gráfico de barras para columnas categóricas
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(col, title=short_name),  # Usar el nombre truncado
                y='count()',
                tooltip=[col, 'count()']
            ).properties(
                width=600,
                height=400,
                title=f"Frecuencia de {short_name}"  # Usar el nombre truncado en el título
            )
        # Mostrar el gráfico
        st.altair_chart(chart, use_container_width=True)

# Función para mostrar correlaciones
def show_correlations(df):
    st.write("Mapa de calor de correlaciones:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

# Función para gráficos personalizados
def custom_plots(df):
    st.write("Gráficos personalizados:")
    with st.form(key="custom_plots_form"):
        col1 = st.selectbox(
            "Selecciona la columna para el eje X",
            df.columns,
            index=0
        )
        col2 = st.selectbox(
            "Selecciona la columna para el eje Y",
            df.columns,
            index=1
        )
        plot_type = st.selectbox(
            "Selecciona el tipo de gráfico",
            ["line", "bar", "scatter", "cross_tab"],
            index=0
        )
        submit_button = st.form_submit_button(label="Go")
    if submit_button:
        st.write(f"Generando gráfico: {plot_type} con {col1} (X) y {col2} (Y)")
        try:
            if plot_type == "cross_tab":
                if not pd.api.types.is_object_dtype(df[col1]) or not pd.api.types.is_object_dtype(df[col2]):
                    st.error("Para gráficos de tabla cruzada, ambas columnas deben ser categóricas.")
                else:
                    plt.figure(figsize=(12, 6))
                    sns.countplot(data=df, x=col1, hue=col2, palette='Set2')
                    plt.title(f'Distribución de {col1} por {col2}')
                    plt.xlabel(col1)
                    plt.ylabel('Frecuencia')
                    plt.xticks(rotation=45)
                    plt.legend(title=col2, loc='upper right')
                    plt.tight_layout()
                    st.pyplot(plt)
            elif plot_type == "line":
                if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                    st.error("Para gráficos de línea, ambas columnas deben ser numéricas.")
                else:
                    chart = alt.Chart(df).mark_line().encode(
                        x=col1,
                        y=col2
                    ).properties(
                        width=600,
                        height=400,
                        title=f"Gráfico de línea: {col1} vs {col2}"
                    )
                    st.altair_chart(chart, use_container_width=True)
            elif plot_type == "bar":
                if not pd.api.types.is_numeric_dtype(df[col2]):
                    st.error("Para gráficos de barras, la columna Y debe ser numérica.")
                else:
                    chart = alt.Chart(df).mark_bar().encode(
                        x=col1,
                        y=f"mean({col2})",
                        color=col1
                    ).properties(
                        width=600,
                        height=400,
                        title=f"Gráfico de barras: {col1} vs {col2}"
                    )
                    st.altair_chart(chart, use_container_width=True)
            elif plot_type == "scatter":
                if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                    st.error("Para gráficos de dispersión, ambas columnas deben ser numéricas.")
                else:
                    chart = alt.Chart(df).mark_circle().encode(
                        x=col1,
                        y=col2,
                        tooltip=[col1, col2]
                    ).properties(
                        width=600,
                        height=400,
                        title=f"Gráfico de dispersión: {col1} vs {col2}"
                    )
                    st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar el gráfico: {e}")
            st.write(f"Detalles del error: {str(e)}")

def analyze_verbatims(df, verbatim_col):
    st.write("##### Procesa tus verbatims")

    # Crear columnas solo para los botones
    col1, col2, col3, col4, col5 = st.columns(5)  # Ajusta el número de columnas según la cantidad de botones

    # Colocar cada botón en una columna
    with col1:
        clean_button = st.button("Clean Verbatims", key="clean_verbatims")

    with col2:
        wordcloud_button = st.button("Plot Wordcloud", key="wordcloud")

    with col3:
        keywords_button = st.button("Frequent Words", key="keywords")

    with col4:
        classify_topic_button = st.button("Classify by Topic", key="classify_topic")

    with col5:
        classify_sentiment_button = st.button("Classify by Sentiment", key="classify_sentiment")

    # Ejecutar la acción correspondiente al botón presionado
    if clean_button:
        df = clean_verbatims(df, verbatim_col)
        st.session_state.df = df  # Actualizar el DataFrame en el estado de sesión
        st.success("Verbatims limpiados correctamente.")

    if wordcloud_button:
        generate_wordcloud(df, verbatim_col)

    if keywords_button:
        show_keywords(df, verbatim_col)

    if classify_topic_button:
        df = classify_by_topic(df, verbatim_col)
        st.session_state.df = df  # Actualizar el DataFrame en el estado de sesión

    if classify_sentiment_button:
        classify_by_sentiment(df, verbatim_col)

    return df

def clean_verbatims(df, verbatim_col):
    df = df.applymap(lambda x: unidecode(str(x)) if isinstance(x, str) else x)
    df[f'{verbatim_col}_clean'] = df[verbatim_col].copy()
    df[f'{verbatim_col}_clean'] = df[f'{verbatim_col}_clean'].apply(preprocess_text)
    return df

def preprocess_text(text):
    if isinstance(text, str) and text.strip():
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('spanish'))
        doc = nlp(' '.join(tokens))
        tokens = [token.lemma_ for token in doc if token.text not in stop_words]
        return ' '.join(tokens)
    return None

def generate_wordcloud(df, verbatim_col):
    all_words = ' '.join(df[f'{verbatim_col}'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def show_keywords(df, verbatim_col):
    all_words = ' '.join(df[f'{verbatim_col}'].dropna())
    word_counts = Counter(all_words.split())
    word_freq_df = pd.DataFrame(word_counts.items(), columns=["Palabra", "Frecuencia"]).sort_values(by="Frecuencia", ascending=False).head(30)
    fig, ax = plt.subplots()
    sns.barplot(x="Frecuencia", y="Palabra", data=word_freq_df, palette="viridis", ax=ax)
    st.pyplot(fig)

def classify_by_topic(df, verbatim_col):
    filtered_df = df[df[f'{verbatim_col}'].notna()].copy()
    model = BERTopic()
    topics, probs = model.fit_transform(filtered_df[f'{verbatim_col}'])
    filtered_df['topics'] = topics
    topic_terms = model.get_topics()
    grouped_verbatims = filtered_df.groupby('topics')[verbatim_col].apply(
        lambda x: x.sample(min(len(x), 20)).tolist()
    ).to_dict()
    def generate_topic_name(topic_id, verbatims):
        terms = [term for term, _ in topic_terms.get(topic_id, [])]
        prompt = f"""Estás analizando respuestas de una encuesta y dando un nombre al tema del que hablan estos verbatims.
        Aquí tienes 20 respuestas agrupadas bajo el mismo tema, junto con algunas palabras clave generadas por un modelo de topic modeling:
        - Palabras clave del modelo: {', '.join(terms[:5])}
        - Ejemplos de respuestas:
        {verbatims}
        Basándote en esto, genera un nombre breve, conciso y claro para el tema que represente bien su contenido.
        Solo responde con el nombre del tema, sin añadir nada más.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Eres un asistente experto en análisis de encuestas."},
                          {"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error al generar el nombre del tópico: {e}")
            return f"Tópico {topic_id}"
    topic_names = {
        topic_id: generate_topic_name(topic_id, verbatims)
        for topic_id, verbatims in grouped_verbatims.items()
    }
    filtered_df['topic_labels'] = filtered_df['topics'].map(topic_names)
    df = df.merge(filtered_df[['topics', 'topic_labels']], how='left', left_index=True, right_index=True)
    if 'topics' in df.columns and 'topic_labels' in df.columns:
        st.success("Las columnas con las temáticas se han guardado correctamente")
    else:
        st.error("Error: Las columnas no se han guardado correctamente.")
    df_filtered = df[df['topics'].notna()]
    topic_counts = df_filtered['topic_labels'].value_counts().reset_index()
    topic_counts.columns = ['Topic Label', 'Count']
    st.write("### Distribución de Temas Refinados")
    st.dataframe(topic_counts)
    st.write("### Gráfica de Frecuencia de Verbatims por Temática")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Topic Label', data=topic_counts, ax=ax, palette="viridis")
    plt.title('Distribución de Temas Refinados')
    plt.xlabel('Frecuencia')
    plt.ylabel('Tema')
    st.pyplot(fig)
    return df

# Modelo de análisis de sentimiento con BERT Multilingual
sentiment_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

def classify_by_sentiment(df, verbatim_col):
    sentiment_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    df['sentiment'] = df[verbatim_col].apply(analyze_sentiment)
    st.write("Distribución de Sentimientos:")
    st.bar_chart(df['sentiment'].value_counts())

def analyze_sentiment(text):
    if isinstance(text, str) and text.strip():
        sentiment = sentiment_model(text[:512])[0]['label']
        return 'positive' if '5' in sentiment or '4' in sentiment else 'negative' if '1' in sentiment or '2' in sentiment else 'neutral'
    return 'neutral'

def main():
    # Inicializar el estado de sesión si no existe
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Menú de navegación con streamlit-option-menu
    with st.sidebar:
        choice = option_menu(
            "Survey Explorer",
            ["Inicio", "Cargar Dataset", "Análisis Exploratorio", "Análisis Gráfico", "Verbatims", "Download Dataset"],
            icons=["house", "upload", "bar-chart", "graph-up", "chat", "download"],
            menu_icon="list",
            default_index=0,
        )

    # Botón de refrescar en todas las secciones
    if choice != "Inicio":
       if st.button("Refrescar"):
            # Forzar a Streamlit a volver a renderizar la sección actual
            if 'refresh' not in st.session_state:
                st.session_state.refresh = True
            else:
                st.session_state.refresh = not st.session_state.refresh
    
    
    if choice == "Inicio":
        st.write("## **🌟 Bienvenido a Survey Explorer 🌟**")
        st.write("Esta herramienta está diseñada para ayudarte a explorar, visualizar y entender tus datos de manera sencilla y eficiente. Carga tu dataset en formato Excel y descubre todo lo que puedes hacer:")  
        st.write("---")  # Línea separadora

        st.write("### **📊 Análisis Exploratorio**")  
        st.write("- **👀 Ver las primeras filas**: Echa un vistazo rápido a tus datos.")  
        st.write("- **📂 Información de columnas**: Descubre el tipo de datos que contiene cada columna.")  
        st.write("- **✏️ Cambiar nombres de columnas**: Personaliza los nombres para que sean más claros.")  

        st.write("---")  # Línea separadora

        st.write("### **📈 Análisis Gráfico**")  
        st.write("- **📉 Gráficos básicos**: Visualiza todas las columnas en gráficos de barras.")  
        st.write("- **🔥 Mapa de calor de correlaciones**: Identifica relaciones entre variables numéricas.")  
        st.write("- **🎨 Gráficos personalizados**: Crea gráficos de línea, barras, dispersión y más.")  

        st.write("---")  # Línea separadora

        st.write("### **💬 Análisis de Verbatims**")  
        st.write("- **🧹 Limpieza de texto**: Preprocesa y limpia las respuestas abiertas.")  
        st.write("- **☁️ Nube de palabras**: Visualiza las palabras más frecuentes.")  
        st.write("- **🏷️ Clasificación por temas**: Usa IA para identificar y etiquetar temas en las respuestas.")  
        st.write("- **😃 Análisis de sentimiento**: Detecta si las respuestas son positivas, negativas o neutras.")  

        st.write("---")  # Línea separadora

        st.write("### **📥 Descarga de resultados**")  
        st.write("- **💾 Guarda tu dataset**: Descarga el archivo con todos los cambios y análisis aplicados.")  

        st.write("---")  # Línea separadora

        st.write("**¡Comienza ahora mismo cargando tu dataset y descubre insights valiosos en tus datos!**")  

        st.write("---")  # Línea separadora

        st.write("### **Cómo usar la aplicación:**")  
        st.write("1. **Carga tu archivo Excel** en la sección 'Cargar Dataset'.")  
        st.write("2. **Navega por las secciones** del menú para explorar, graficar y analizar tus datos.")  
        st.write("3. **Descarga los resultados** cuando hayas terminado.")  

        st.write("---")  # Línea separadora

        st.write("### **🚀 ¿Listo para empezar?**")  
        st.write("¡Sube tu archivo y comienza a explorar!")  


    elif choice == "Cargar Dataset":
        st.write("### 🚀 Sube aquí tu Dataset para comenzar")
        df = upload_file()
        if df is not None:
            st.session_state.df = df
            st.success("Tu Dataset se ha cargado correctamente")

    elif choice == "Análisis Exploratorio":
        st.write("### 👀 Análisis Exploratorio")
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            tab1, tab2, tab3 = st.tabs(["Sample", "Tipo de datos", "Cambia los nombres"])
            with tab1:
                show_sample(df)
            with tab2:
                show_columns(df)
            with tab3:
                df = change_column_names(df)
                st.session_state.df = df  # Guardar cambios en el estado de sesión
        else:
            st.warning("Por favor, carga un dataset primero.")

    elif choice == "Análisis Gráfico":
        st.write("### 📈 Análisis Gráfico")
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            tab1, tab2, tab3 = st.tabs(["Visualiza tus columnas", "Correlaciones", "Custom Plots"])
            with tab1:
                plot_data(df)
            with tab2:
                show_correlations(df)
            with tab3:
                custom_plots(df)
        else:
            st.warning("Por favor, carga un dataset primero.")

    elif choice == "Verbatims":
        st.write("### 💬 Análisis de Verbatims")
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            verbatim_col = st.selectbox("Selecciona la columna de verbatims", df.columns, key="verbatim_col")
            df = analyze_verbatims(df, verbatim_col)
            st.session_state.df = df  # Guardar cambios en el estado de sesión
        else:
            st.warning("Por favor, carga un dataset primero.")

    elif choice == "Download Dataset":
        st.write("### Descargar Dataset")
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Datos')
            output.seek(0)
            st.download_button(
                label="Descargar Dataset",
                data=output,
                file_name="dataset_modificado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Por favor, carga un dataset primero.")

if __name__ == "__main__":
    main()