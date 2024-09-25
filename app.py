import streamlit as st
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# Load environment variables for API keys
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google Generative AI
genai.configure(api_key=google_api_key)

# 1. Initialize Pinecone for vector search
pinecone.init(api_key=pinecone_api_key, environment="us-east-1")
index_name = "mindmendai"

# If index does not exist, create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # Dimension of SentenceTransformer embeddings

index = pinecone.Index(index_name)

# 2. Initialize Sentence Transformer for embedding text
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models from sentence-transformers

# 3. Define function to extract and chunk PDF content
def extract_and_chunk_pdfs(pdf_files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for pdf_file in pdf_files:
        text = ""
        with open(pdf_file, "rb") as file:
            # Read binary data from the file
            binary_pdf_data = file.read()

            # Manually decode portions of the PDF
            # PDF text is often stored between `BT` (Begin Text) and `ET` (End Text) markers
            try:
                # Decoding the binary content
                text_data = binary_pdf_data.decode('latin1')  # 'latin1' decoding used here for simplicity
            except UnicodeDecodeError:
                text_data = binary_pdf_data.decode('utf-8', errors='ignore')  # Ignore problematic bytes

            # Extract text between BT and ET markers
            in_text_object = False
            current_text = []
            
            for line in text_data.splitlines():
                if "BT" in line:
                    in_text_object = True
                if in_text_object:
                    # Extract text enclosed in parentheses ( )
                    if "(" in line and ")" in line:
                        start = line.find("(")
                        end = line.find(")")
                        if start != -1 and end != -1:
                            current_text.append(line[start + 1:end])

                if "ET" in line:
                    in_text_object = False

            # Join the extracted text and append to main text
            text += " ".join(current_text)

        # Split text into chunks using the text splitter
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    return all_chunks


# 4. Preprocess and index PDF chunks into Pinecone
def index_pdf_content(pdf_files):
    pdf_text_chunks = extract_and_chunk_pdfs(pdf_files)
    for chunk in pdf_text_chunks:
        vector = embedding_model.encode(chunk)  # Get vector using SentenceTransformer
        index.upsert([(chunk, vector)])  # Indexing each chunk

# 5. Preload your PDF files (ensure files are placed in a directory for processing)
pdf_files = ["sodapdf-converted.pdf"]
index_pdf_content(pdf_files)

# 6. Set up prompt template with psychologist's tone
prompt_template = """
You are a psychologist assisting a user with mental health queries. Your responses should be based on the information from the provided documents. Use a supportive and empathetic tone, offering professional guidance while being considerate.

Question: {question}

Answer in a calm, empathetic manner:
"""
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# 7. Create function to interact with Google Generative AI model
def query_generative_ai(question):
    model = "models/chat-bison-001"  # Choose the Google Generative AI model
    response = genai.chat(
        model=model,
        messages=[{"content": question}],
        temperature=0.3
    )
    return response["candidates"][0]["content"]

# 8. Initialize retriever using Pinecone
class PineconeRetriever:
    def __init__(self, index, embedding_fn):
        self.index = index
        self.embedding_fn = embedding_fn

    def retrieve(self, query):
        # Embed the query and retrieve results from Pinecone
        query_vector = self.embedding_fn(query)
        results = self.index.query(queries=[query_vector], top_k=5)
        return [result["metadata"]["text"] for result in results["matches"]]

retriever = PineconeRetriever(index, embedding_model.encode)

# 9. Set up Conversational Retrieval Chain for chat with memory
def conversational_chain(question, chat_history):
    # Retrieve relevant chunks from Pinecone
    retrieved_docs = retriever.retrieve(question)

    # Construct a conversational context
    context = "\n".join(retrieved_docs)
    final_prompt = prompt.format(question=question)

    # Query Google Generative AI with context and question
    ai_response = query_generative_ai(final_prompt)

    # Update chat history
    chat_history.append({"question": question, "answer": ai_response})

    return ai_response, chat_history

# 10. Streamlit Interface: Mental Health Chatbot Application
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("Mental Health Chatbot 🧠💬")
st.write("Ask any mental health-related questions. The chatbot will provide responses based on preloaded documents in a psychologist's tone.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for querying the chatbot
user_query = st.text_input("Ask your question here:", "")

# When user clicks the "Submit" button
if st.button("Submit"):
    if user_query:
        # Query the chatbot and get the response
        response, updated_chat_history = conversational_chain(user_query, st.session_state.chat_history)
        
        # Update session state with new chat history
        st.session_state.chat_history = updated_chat_history
        
        # Display chatbot's response
        st.markdown(f"**Chatbot Response**: {response}")
    else:
        st.warning("Please enter a question.")

# Display the chat history, if available
if st.session_state.chat_history:
    st.write("### Chat History")
    for i, message in enumerate(st.session_state.chat_history):
        st.markdown(f"**User**: {message['question']}")
        st.markdown(f"**Chatbot**: {message['answer']}")
