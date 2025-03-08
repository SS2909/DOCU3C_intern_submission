from flask import Flask, render_template, request, jsonify
import os
import pdfminer.high_level
import spacy
from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nli_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

def extract_text_with_page_numbers(pdf_path):
    """Extracts text from a PDF with page-wise references."""
    texts = pdfminer.high_level.extract_text(pdf_path).split("\f")
    page_text = {i + 1: text.strip() for i, text in enumerate(texts) if text.strip()}  # Page number starts from 1
    return page_text

def classify_argument(sentence):
    """Classifies a sentence as supporting ('for') or opposing ('against') using NLI."""
    labels = ["for", "against"]
    result = nli_classifier(sentence, candidate_labels=labels)
    
    # Assign based on highest confidence
    label = result["labels"][0]  # First label has the highest confidence
    confidence = result["scores"][0]

    # Threshold for strong classification
    return label if confidence > 0.5 else None  # Discard low-confidence results

def analyze_text_with_page_numbers(page_texts):
    """Analyzes the text and extracts arguments with page references."""
    key_arguments = {"for": [], "against": []}
    key_entities = set()
    summaries = []

    for page_num, text in page_texts.items():
        doc = nlp(text)

        # Extract legal-related entities
        key_entities.update([ent.text for ent in doc.ents if ent.label_ in ["LAW", "ORG", "GPE"]])

        # Summarization (No max length restriction)
        if len(text.split()) > 50:  # Summarize only if text is meaningful
            summary = summarizer(text, min_length=30, do_sample=False)[0]['summary_text']
        else:
            summary = text  # If text is too short, return it as is

        summaries.append(f"Page {page_num}: {summary}")

        # Classify arguments using ML-based approach
        for sent in doc.sents:
            sentence = sent.text.strip()
            classification = classify_argument(sentence)

            if classification == "for":
                key_arguments["for"].append({"argument": sentence, "page": page_num})
            elif classification == "against":
                key_arguments["against"].append({"argument": sentence, "page": page_num})

    # Pick top 10 arguments (5 for each side)
    top_for = key_arguments["for"][:5]
    top_against = key_arguments["against"][:5]

    return {
        "summaries": summaries,
        "key_entities": list(key_entities),
        "arguments_for": top_for,
        "arguments_against": top_against
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')  # Ensure correct field name
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Extract and analyze text
    page_texts = extract_text_with_page_numbers(filepath)
    analysis = analyze_text_with_page_numbers(page_texts)

    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)
