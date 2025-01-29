import streamlit as st
import pandas as pd
import PyPDF2 
import joblib  
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns  
import os
from docx import Document  
# Load the model and vectorizer
model = joblib.load("C:\\Users\\LENOVO\\Desktop\\DS-P490(GROUP-5)\\modelDT.pkl")  # Update with your model's filename
vectorizer = joblib.load("C:\\Users\\LENOVO\\Desktop\\DS-P490(GROUP-5)\\vector.pkl")  # Update with your vectorizer's filename

# App Title
st.set_page_config(page_title="AI-Powered Resume Classification", layout="wide")
st.title("SmartHire-AI-Powered Resume Classification and Analytics Hub")

# Sidebar Header
st.sidebar.header("Options")

# Tabs for Organized Content
tabs = st.tabs(["Classification", "Visualizations", "Feedback", "Insights"])

# File Upload with History
uploaded_files = []
uploaded_file = st.file_uploader("Upload a Resume (PDF/Text/Word)", type=["pdf", "txt", "docx"], key="uploader")

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

if uploaded_file:
    try:
        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() for page in reader.pages)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        uploaded_files.append({"filename": uploaded_file.name, "text": text})

        # Classification
        with tabs[0]:
            st.subheader("Resume Classification")
            st.text_area("Extracted Text", text, height=200)
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            confidence_scores = model.predict_proba(X)[0]

            st.write(f"### Predicted Category: **{prediction}**")
            st.write("#### Confidence Scores:")
            confidence_df = pd.DataFrame({
                "Category": model.classes_,
                "Confidence": confidence_scores
            })
            st.table(confidence_df)

            # Word Cloud
            st.write("### Word Cloud")
            wordcloud = WordCloud(background_color="white", colormap="coolwarm", max_words=50).generate(text)
            st.image(wordcloud.to_array(), use_column_width=True)

            # Download Classification Result
            result_data = {
                "Extracted Text": text,
                "Predicted Category": prediction
            }
            result_df = pd.DataFrame([result_data])
            st.download_button(
                label="Download Classification Result",
                data=result_df.to_csv(index=False),
                file_name="classification_result.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Visualizations
with tabs[1]:
    st.subheader("Dataset Visualizations")

    # Example dataset
    data = pd.DataFrame({
        "Category": ["IT", "Finance", "Healthcare", "Education"],
        "Count": [100, 80, 120, 50]
    })

    st.write("### Dataset Table")
    st.table(data)

    st.write("### Enhanced Bar Chart")
    fig, ax = plt.subplots()
    sns.barplot(x="Category", y="Count", data=data, palette="coolwarm", ax=ax)
    ax.set_title("Category Distribution", fontsize=16)
    st.pyplot(fig)

    st.write("### Enhanced Pie Chart")
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(data["Count"], labels=data["Category"], autopct="%1.1f%%", startangle=90, colors=sns.color_palette("coolwarm", len(data)))
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Category Breakdown", fontsize=16)
    ax.axis("equal")
    st.pyplot(fig)

# Feedback
with tabs[2]:
    st.subheader("Feedback Section")
    feedback = st.text_area("Let us know if the classification was incorrect or if you have suggestions.")
    if st.button("Submit Feedback"):
        feedback_path = "feedback.csv"
        if os.path.exists(feedback_path):
            feedback_df = pd.read_csv(feedback_path)
        else:
            feedback_df = pd.DataFrame(columns=["Feedback"])

        feedback_df = feedback_df.append({"Feedback": feedback}, ignore_index=True)
        feedback_df.to_csv(feedback_path, index=False)
        st.success("Thank you for your feedback!")

        st.write("### Feedback Overview")
        st.table(feedback_df.tail(5))

# Insights Tab
with tabs[3]:
    st.subheader("Insights and Analytics")

    # Top Categories
    st.write("### Most Common Categories")
    top_categories = data.sort_values(by="Count", ascending=False).head(3)
    st.table(top_categories)

    st.write("### Heatmap of Category Counts")
    fig, ax = plt.subplots()
    sns.heatmap(data.set_index("Category").T, annot=True, fmt="d", cmap="coolwarm", cbar=False, ax=ax)
    st.pyplot(fig)

    st.write("### Custom Metrics")
    st.info("Metrics can be extended here, such as precision, recall, or F1-score analytics.")
