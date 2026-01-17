import joblib
import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- App Title ----------
st.set_page_config(page_title="Bird Text Clustering Mini App", layout="wide")
st.title("🦜 Bird Text Clustering Mini App")
st.write("Paste a bird-related description and the model will assign it to a discovered cluster.")

# ---------- Load artifacts ----------
vectorizer = joblib.load("vectorizer.joblib")
svd = joblib.load("svd200.joblib")
kmeans = joblib.load("kmeans.joblib")
cluster_keywords = joblib.load("cluster_keywords.joblib")
df = pd.read_csv("birds_text_with_clusters_FINAL.csv")

# ---------- Same preprocessing (must match notebook) ----------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
domain_stopwords = {"bird", "birds", "species", "genus", "family", "taxon", "taxonomy", "abstract"}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if (t not in stop_words)
        and (t not in domain_stopwords)
        and (len(t) > 2)
    ]
    return " ".join(tokens)

# ---------- Optional cluster names (edit these if you want) ----------
cluster_names = {
    0: "Tropical & Subtropical Bird Habitats",
    1: "General & Taxonomic Bird Descriptions",
}

# ---------- UI ----------
user_text = st.text_area("Enter text", height=160, placeholder="Example: This bird lives in tropical moist montane forests...")

col1, col2 = st.columns([1, 1])

if st.button("Predict Cluster"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        clean = preprocess_text(user_text)
        vec = vectorizer.transform([clean])
        vec_200 = svd.transform(vec)
        pred = int(kmeans.predict(vec_200)[0])

        with col1:
            st.subheader("Prediction")
            st.write(f"**Cluster:** {pred}")
            st.write(f"**Label:** {cluster_names.get(pred, f'Cluster {pred}')}")

            st.subheader("Top Keywords")
            kws = cluster_keywords.get(pred, [])
            st.write(", ".join(kws[:15]) if kws else "No keywords available.")

        with col2:
            st.subheader("Representative Examples (from your dataset)")
            # show 3 example texts from that cluster
            examples = df[df["cluster"] == pred]["text"].head(3).tolist()
            if not examples:
                st.write("No examples found.")
            else:
                for i, ex in enumerate(examples, 1):
                    st.markdown(f"**Example {i}:** {ex[:450]}...")

st.divider()
st.caption("This app uses your trained TF-IDF + TruncatedSVD + KMeans pipeline (no retraining).")
