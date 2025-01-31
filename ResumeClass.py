import streamlit as st
import pandas as pd
import PyPDF2  # For extracting text from PDFs
import joblib  # Corrected joblib import
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import docx  # For extracting text from DOCX files

# Load the trained model and vectorizer
model = joblib.load("modelDT.pkl")
vectorizer = joblib.load("vector.pkl")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# App Title
st.set_page_config(page_title="Resume Classification Dashboard", layout="wide")
st.title("📄 Resume Classification Dashboard")

# Sidebar Header
st.sidebar.header("Navigation")
tabs = st.sidebar.radio("Go to", ["Upload & Classify", "Visualizations", "Insights"])

# File Upload
uploaded_file = st.file_uploader("Upload a Resume (PDF/Text/DOCX)", type=["pdf", "txt", "docx"])

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join(page.extract_text() for page in reader.pages)
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

if tabs == "Upload & Classify":
    st.subheader("Upload & Classify Resume")
    if uploaded_file:
        try:
            # Process the uploaded file
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")

            st.write("### Resume Preview:")
            st.text_area("Extracted Text", text, height=200)

            # NLP Processing
            doc = nlp(text)
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Vectorization & Prediction
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            confidence_scores = model.predict_proba(X)[0]

            st.write(f"### 🎯 Predicted Category: **{prediction}**")
            st.write("#### Confidence Scores:")
            confidence_df = pd.DataFrame({
                "Category": model.classes_,
                "Confidence": confidence_scores
            })
            st.table(confidence_df)

            # Named Entity Recognition (NER)
            st.write("### 🔍 Named Entity Recognition (NER)")
            for ent, label in named_entities:
                st.markdown(f"✅ **{ent}** - `{label}`")

            # Word Cloud
            st.write("### 📊 Word Cloud")
            wordcloud = WordCloud(background_color="white").generate(text)
            st.image(wordcloud.to_array(), use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif tabs == "Visualizations":
    st.subheader("Data Visualizations")
    data = pd.DataFrame({
        "Category": ["IT", "Finance", "Healthcare", "Education"],
        "Count": [100, 80, 120, 50]
    })

    st.write("### 📊 Category Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x="Category", y="Count", data=data, palette="coolwarm", ax=ax)
    ax.set_title("Category Distribution", fontsize=16)
    st.pyplot(fig)

    st.write("### 🥧 Category Breakdown")
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(data["Count"], labels=data["Category"], autopct="%1.1f%%", startangle=90)
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Category Breakdown", fontsize=16)
    st.pyplot(fig)

elif tabs == "Insights":
    st.subheader("Resume Data Insights")
    st.write("### 🔥 Most Frequent Words")
    sample_texts = ["data analysis python", "machine learning", "SQL database management"]
    word_freq = pd.Series(" ".join(sample_texts).split()).value_counts().head(10)
    st.bar_chart(word_freq)

    st.write("### 🔬 Feature Importance (Placeholder)")
    st.info("Add feature importance charts here in the future.")
