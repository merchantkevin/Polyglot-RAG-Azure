import os
import uuid
import requests
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType, 
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile
)

# Load environment variables from your .env file
load_dotenv()

# Setup external OpenAI Client
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Azure Configurations
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
INDEX_NAME = "multilingual-portfolio-index"

def extract_and_chunk_pdf(file_path, chunk_size=1000):
    print(f"Reading and chunking {file_path}...")
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() for page in reader.pages])
    
    # Simple character chunking to respect API limits
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def translate_to_english(text_chunk):
    print("Translating chunk via Azure F0 Translator...")
    url = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=en"
    headers = {
        'Ocp-Apim-Subscription-Key': TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': TRANSLATOR_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text_chunk}] 
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status() # Will throw an error if auth fails
    return response.json()[0]['translations'][0]['text']

def get_embedding(text):
    print("Generating embedding via external OpenAI (text-embedding-3-small)...")
    response = OPENAI_CLIENT.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def setup_search_index():
    print("Configuring Azure AI Search F0 Index...")
    credential = AzureKeyCredential(SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    
    # Define Vector Search Config for OpenAI's 1536 dimensions
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myProfile", algorithm_configuration_name="myHnsw")]
    )
    
    # Define the schema (columns) for our database
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myProfile")
    ]
    
    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)

def push_to_search(search_client, document_list):
    print(f"Pushing {len(document_list)} chunks to Azure AI Search...")
    search_client.upload_documents(documents=document_list)
    print("Pipeline Complete! All documents indexed successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    pdf_filename = "sample_es.pdf"
    
    if not os.path.exists(pdf_filename):
        print(f"ERROR: Please place a file named '{pdf_filename}' in this folder.")
        exit(1)
        
    chunks = extract_and_chunk_pdf(pdf_filename)
    search_client = setup_search_index()
    
    documents_to_upload = []
    
    for chunk in chunks:
        # 1. Translate
        english_text = translate_to_english(chunk)
        # 2. Embed
        vector = get_embedding(english_text)
        # 3. Format for Database
        documents_to_upload.append({
            "id": str(uuid.uuid4()),
            "content": english_text,
            "content_vector": vector
        })
        
    # Push all processed chunks to the cloud
    push_to_search(search_client, documents_to_upload)