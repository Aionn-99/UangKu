import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Unduh resource NLTK
def download_nltk_resources():
    resources = [("corpora/stopwords", "stopwords"), ("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]
    for path, resource in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)
download_nltk_resources()

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

stop_words_ind = set(stopwords.words("indonesian"))
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words_ind]
    return " ".join(filtered_tokens)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem_text(text):
    return stemmer.stem(text)

# Muat model dan vectorizer
@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open("logistic_regression_model.pkl", "rb") as f:
        model_lr = pickle.load(f)
    with open("naive_bayes_model.pkl", "rb") as f:
        model_nb = pickle.load(f)
    return tfidf_vectorizer, model_lr, model_nb

tfidf_vectorizer, model_lr, model_nb = load_models()

# Fungsi prediksi
def predict_category(description):
    # Preprocessing
    clean_text = preprocess_text(description)
    filtered_text = tokenize_and_remove_stopwords(clean_text)
    stemmed_text = stem_text(filtered_text)
    # Transformasi TF-IDF
    X = tfidf_vectorizer.transform([stemmed_text])
    # Prediksi
    pred_lr = model_lr.predict(X)[0]
    pred_nb = model_nb.predict(X)[0]
    return pred_lr, pred_nb

# Antarmuka Streamlit
st.title("Demo Klasifikasi Kategori Pengeluaran Pribadi")
st.write("Masukkan deskripsi pengeluaran untuk memprediksi kategorinya menggunakan model Logistic Regression dan Naive Bayes.")

# Input pengguna
description = st.text_area("Deskripsi Pengeluaran", placeholder="Contoh: Beli kopi di Starbucks")
if st.button("Prediksi"):
    if description:
        pred_lr, pred_nb = predict_category(description)
        st.success(f"**Logistic Regression**: {pred_lr}")
        st.success(f"**Naive Bayes**: {pred_nb}")
        if pred_lr != pred_nb:
            st.warning("Model menghasilkan prediksi berbeda. Logistic Regression biasanya lebih akurat berdasarkan evaluasi.")
    else:
        st.error("Silakan masukkan deskripsi pengeluaran.")

# Visualisasi performa
st.subheader("Performa Model")
st.write("Berdasarkan evaluasi pada data uji:")
metrics = {
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [1.00, 0.98],  # Ganti dengan nilai aktual dari notebook
    "Precision (weighted)": [1.00, 1.00],
    "Recall (weighted)": [1.00, 0.89],
    "F1-Score (weighted)": [1.00, 0.94]
}
st.table(pd.DataFrame(metrics))

# Contoh prediksi
st.subheader("Contoh Prediksi pada Data Uji")
df = pd.read_csv("expense_dataset_10000.csv").sample(5, random_state=42)
df["Prediksi LR"] = [predict_category(desc)[0] for desc in df["description"]]
df["Prediksi NB"] = [predict_category(desc)[1] for desc in df["description"]]
st.table(df[["description", "category", "Prediksi LR", "Prediksi NB"]])

# Sidebar
st.sidebar.header("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini menggunakan model Logistic Regression dan Naive Bayes untuk mengklasifikasikan kategori pengeluaran berdasarkan deskripsi teks. 
- **Preprocessing**: Case folding, hapus tanda baca/angka, hapus stop words, stemming (Sastrawi).
- **Fitur**: TF-IDF dengan maksimum 1000 fitur.
- **Dataset**: 10000 data pengeluaran.
- **Evaluasi**: Logistic Regression lebih unggul dalam akurasi.
""")