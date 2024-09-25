from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-pro")

# Initialize Flask app
app = Flask(__name__)

# Function to load PDF data manually
def load_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    # Manually decode PDF content (this is a simplistic approach)
    text = ""
    for i in range(len(pdf_content)):
        if pdf_content[i] == 10:  # Line Feed
            text += '\n'
        elif 32 <= pdf_content[i] <= 126:  # Printable ASCII characters
            text += chr(pdf_content[i])
    
    return text

# Generate embeddings for the document
def generate_embeddings(text):
    sentences = text.split('\n')  # Split text into sentences
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return sentences, embeddings, model  # return the model as well

# Find the most relevant sentence based on the user's question
def find_relevant_sentence(question, sentences, embeddings, model):
    # Encode the question using the same model
    question_embedding = model.encode([question])
    # Compute cosine similarity between question and document embeddings
    similarities = np.dot(embeddings, question_embedding.T).squeeze()
    best_index = np.argmax(similarities)
    return sentences[best_index]

# Load and process PDF data
pdf_path = "sodapdf-converted.pdf"  # Path to your custom data PDF file
pdf_text = load_pdf(pdf_path)
sentences, embeddings, embedding_model = generate_embeddings(pdf_text)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    
    # Find the most relevant sentence based on the question
    relevant_sentence = find_relevant_sentence(user_input, sentences, embeddings, embedding_model)

    # Get response from Gemini model
    chat = gemini_model.start_chat(history=[])
    response = chat.send_message(f"{relevant_sentence}\n{user_input}", stream=True)

    # Prepare the response to send back
    response_text = ""
    for chunk in response:
        response_text += chunk.text

    return jsonify({'user_input': user_input, 'response_text': response_text})

if __name__ == "__main__":
    app.run(debug=True)
