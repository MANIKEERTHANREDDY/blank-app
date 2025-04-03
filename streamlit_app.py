import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import google.generativeai as genai 
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Google Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key is None:
    raise ValueError("No API key found in environment variables")

genai.configure(api_key=api_key)

# Load the BioBERT NER model
@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="first")

ner_pipeline = load_ner_pipeline()

# Function to merge WordPiece tokens into full words
def detokenize_wordpieces(tokens):
    merged_tokens = []
    current_word = ""

    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                merged_tokens.append(current_word)
            current_word = token

    if current_word:
        merged_tokens.append(current_word)

    return merged_tokens

# Function to get drug recommendations from Google Gemini
def get_drug_recommendation(disease):
    cleaned_disease = disease.replace("-", " ").capitalize()
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"List only the drug names used to treat {cleaned_disease}, separated by commas. No explanations, just drug names."
    )
    return response.text.strip() if response.text else "No recommendations found"

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# List of non-disease terms to exclude
EXCLUDED_TERMS = {"ecg", "troponin", "examination", "sars - cov - 2"}
DISEASE_OVERRIDES = {"hypertension", "type 2 diabetes mellitus"}

# Streamlit UI
st.title("🔬 Biomedical NER (Powered by BioBERT)")
st.markdown(
    "This app is developed using **BioBERT** for **medical research**."
)

# UI Layout
tab1, tab2 = st.tabs(["📄 Analyze Text", "📂 Upload File"])

with tab1:
    text = st.text_area("Enter text for NER analysis:")
    if st.button("🔍 Analyze Text", type="primary"):
        if text.strip():
            results = ner_pipeline(text)
            recognized_diseases = set()
            entities = []

            for entity in results:
                word = detokenize_wordpieces([entity["word"].replace("##", "")])[0]
                entity_type = entity.get("entity", entity.get("entity_group", "Unknown"))
                if word.lower() in DISEASE_OVERRIDES:
                    entity_type = "Disease_disorder"
                if any(keyword in entity_type.lower() for keyword in ["disease", "disorder"]) and word.lower() not in EXCLUDED_TERMS:
                    recognized_diseases.add(word.lower())
                entities.append({"Word": word, "Entity": entity_type})

            if entities:
                st.subheader("🔬 Named Entities Detected:")
                st.table(entities)
            else:
                st.info("No biomedical entities detected.")

            if recognized_diseases:
                st.subheader("💊 Recommended Drugs:")
                for disease in recognized_diseases:
                    drugs = get_drug_recommendation(disease)
                    st.markdown(f"**{disease.title()}**: {drugs}")
            else:
                st.info("No diseases detected.")
        else:
            st.warning("Please enter text for analysis.")

with tab2:
    uploaded_file = st.file_uploader("Upload a text, PDF, or Word file", type=["txt", "pdf", "docx"])
    if st.button("📂 Analyze File", type="primary"):
        if uploaded_file is not None:
            text = extract_text_from_file(uploaded_file)
            if text.strip():
                results = ner_pipeline(text)
                recognized_diseases = set()
                entities = []
                for entity in results:
                    word = detokenize_wordpieces([entity["word"].replace("##", "")])[0]
                    entity_type = entity.get("entity", entity.get("entity_group", "Unknown"))
                    if word.lower() in DISEASE_OVERRIDES:
                        entity_type = "Disease_disorder"
                    if any(keyword in entity_type.lower() for keyword in ["disease", "disorder"]) and word.lower() not in EXCLUDED_TERMS:
                        recognized_diseases.add(word.lower())
                    entities.append({"Word": word, "Entity": entity_type})

                if entities:
                    st.subheader("🔬 Named Entities Detected:")
                    st.table(entities)
                else:
                    st.info("No biomedical entities detected.")

                if recognized_diseases:
                    st.subheader("💊 Recommended Drugs:")
                    for disease in recognized_diseases:
                        drugs = get_drug_recommendation(disease)
                        st.markdown(f"**{disease.title()}**: {drugs}")
                else:
                    st.info("No diseases detected.")
            else:
                st.warning("No text extracted from the file.")
        else:
            st.warning("Please upload a file for analysis.")
