
from flask import Flask, request, jsonify,render_template,Response
import spacy
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
import tempfile
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests


app = Flask(__name__)

CORS(app)
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_similarity(model, source_text, target_texts):
    # Tokenize source and target texts
    source_input = tokenizer(source_text, padding=True, truncation=True, return_tensors='pt')
    target_inputs = tokenizer(target_texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        source_output = model(**source_input)
        target_outputs = model(**target_inputs)

    # Perform pooling for source and target embeddings
    source_embedding = mean_pooling(source_output, source_input['attention_mask'])
    target_embeddings = mean_pooling(target_outputs, target_inputs['attention_mask'])

    # Calculate cosine similarity using sentence-transformers
    cosine_scores = torch.nn.functional.cosine_similarity(source_embedding, target_embeddings).cpu().numpy()

    return cosine_scores


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def preprocess_text(text_to_clean):
    # Remove URLs
    cleaned_text = re.sub('http\S+\s*', ' ', text_to_clean)
    # Remove RT and cc
    cleaned_text = re.sub('RT|cc', ' ', cleaned_text)
    # Remove hashtags
    cleaned_text = re.sub('#\S+', '', cleaned_text)
    # Remove mentions
    cleaned_text = re.sub('@\S+', ' ', cleaned_text)
    # Remove punctuations
    cleaned_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleaned_text)
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7f]', r' ', cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub('\s+', ' ', cleaned_text)
    # Remove leading and trailing whitespaces
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def parsing(path, text):
    nlp = spacy.load(path)
    doc = nlp(text)
    print(doc.ents)
    entity_dict = {}
    for ent in doc.ents:
        if ent.label_ in entity_dict:
            entity_dict[ent.label_].append(ent.text)
        else:
            entity_dict[ent.label_] = [ent.text]
    return entity_dict

def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()

    # Remove extra spaces and new lines
    text = ' '.join(text.split())
    
    return text

def jobClassification(resume):
    
        input_resume = preprocess_text(resume)
        # Load the saved model
        with open('./RankingAssets/nb_classifier_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('./RankingAssets/word_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

        new_data_vectorized = loaded_vectorizer.transform([input_resume])

        # Make predictions on the vectorized new data using the loaded model
        new_predictions = loaded_model.predict(new_data_vectorized)

        # The variable 'new_predictions' now contains the model predictions for the new data
        df = pd.read_csv('./RankingAssets/cv_data.csv')
        label = LabelEncoder()
        df['Category_but_Encoded'] = label.fit_transform(df['Category'])
        predicted_label = label.inverse_transform(new_predictions)

        return predicted_label[0]

    

@app.route('/resumeParser', methods=['POST'])
def parse_resume():
    try:
        if 'resume_pdf' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        resume_pdf = request.files['resume_pdf']

        if resume_pdf.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if resume_pdf and resume_pdf.filename.endswith('.pdf'):
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                file_path = tmp_file.name
                resume_pdf.save(file_path)

            # Extract text from the PDF file
            resume_text = extract_text_from_pdf(file_path)

            # Rest of your code
            nlp = spacy.load('C:/RankerModels/CV_output_model/model-best')

            doc = nlp(resume_text)
            print(doc.ents)

            entity_dict = {}
            for ent in doc.ents:
                if ent.label_ in entity_dict:
                    entity_dict[ent.label_].append(ent.text)
                else:
                    entity_dict[ent.label_] = [ent.text]
            print(entity_dict)
            return jsonify({
                'Domain': jobClassification(resume_text),
                'entities': entity_dict
            })
        else:
            return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobDescriptionParser', methods=['POST'])
def parse_jd():
    try:
        if 'job_pdf' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        job_pdf = request.files['job_pdf']

        if job_pdf.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if job_pdf and job_pdf.filename.endswith('.pdf'):
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                file_path = tmp_file.name
                job_pdf.save(file_path)

            # Extract text from the PDF file
            job_text = extract_text_from_pdf(file_path)

            # Rest of your code
            nlp = spacy.load('C:/RankerModels/jd_train_output_model/model-best')

            doc = nlp(job_text)
            print(doc.ents)

            entity_dict = {}
            for ent in doc.ents:
                if ent.label_ in entity_dict:
                    entity_dict[ent.label_].append(ent.text)
                else:
                    entity_dict[ent.label_] = [ent.text]

            return jsonify({
                'entities': entity_dict
            })
        else:
            return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/jobClassifier', methods=['POST'])
def job_classifier():
    try:
        # Get the input resume from the request
        resume = request.json.get('input_resume')
        input_resume = preprocess_text(resume)
        # Load the saved model
        with open('./RankingAssets/nb_classifier_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('./RankingAssets/word_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

        new_data_vectorized = loaded_vectorizer.transform([input_resume])

        # Make predictions on the vectorized new data using the loaded model
        new_predictions = loaded_model.predict(new_data_vectorized)

        # The variable 'new_predictions' now contains the model predictions for the new data
        df = pd.read_csv('./RankingAssets/cv_data.csv')
        label = LabelEncoder()
        df['Category_but_Encoded'] = label.fit_transform(df['Category'])
        predicted_label = label.inverse_transform(new_predictions)

        return jsonify({'predictions': predicted_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/matchingscore_latest', methods=['POST'])
def jobMatchingScore_latest():
    try:
        # Check if both resume and job description PDF files are provided in the request
        if 'resume_pdf' not in request.files or 'jobdesc_pdf_url' not in request.form:
            return jsonify({'error': 'Both resume PDF file and job description PDF URL are required'}), 400

        resume_pdf = request.files['resume_pdf']
        jobdesc_pdf_url = request.form['jobdesc_pdf_url']

        if resume_pdf.filename == '' or not jobdesc_pdf_url:
            return jsonify({'error': 'No selected file for either resume or job description PDF URL'}), 400

        if resume_pdf.filename.endswith('.pdf'):
            # Download the job description PDF file from the provided URL
            jobdesc_response = requests.get(jobdesc_pdf_url)
            if jobdesc_response.status_code != 200:
                return jsonify({'error': f'Failed to download job description PDF file from the provided URL: {jobdesc_response.status_code}'}), 400
            
            # Extract text from the resume and job description PDF files
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file1, \
                    tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file2:
                resume_pdf.save(tmp_file1.name)
                tmp_file2.write(jobdesc_response.content)
                tmp_file2.seek(0)  # Reset file pointer to the beginning
                resume_text = extract_text_from_pdf(tmp_file1.name)
                jobdesc_text = extract_text_from_pdf(tmp_file2.name)

            # Calculate similarity score between the resume and job description texts (using your calculate_similarity function)
                clean_res=preprocess_text(resume_text)
                clean_jd=preprocess_text(jobdesc_text)

                parsedRes= parsing("C:/RankerModels/CV_output_model/model-best",clean_res)
                parsedjd= parsing("C:/RankerModels/jd_train_output_model/model-best",clean_jd)
            similarity_scores = calculate_similarity(model, str(parsedRes), str(parsedjd))
            print(similarity_scores)
            resume_domain=jobClassification(resume_text)

            return jsonify({
                'score': similarity_scores[0] * 100,
                "predicted_domain":resume_domain
                })

        else:
            return jsonify({'error': 'Uploaded resume file must be in PDF format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# main driver function
if __name__ == '__main__':

	app.run()
