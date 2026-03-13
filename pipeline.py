import os
import uuid
import requests
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

# Load environment variables
load_dotenv()

# --- Configurations ---
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTEL_KEY")

TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "global")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "nexacorp-global-index" # New index name for our enterprise data

# --- Clients ---
doc_intel_client = DocumentAnalysisClient(
    endpoint=DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(DOC_INTEL_KEY)
)
search_index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY)
)
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY)
)

def create_index():
    """Creates a new Azure Search Index that includes a 'source_file' metadata field."""
    print(f"Creating/Updating Azure AI Search Index: {INDEX_NAME}...")
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="source_file", type=SearchFieldDataType.String, filterable=True), # NEW METADATA FIELD
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]
    
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )
    
    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    search_index_client.create_or_update_index(index)
    print("Index ready.\n")

def extract_text_and_tables(file_path):
    """Uses Azure Document Intelligence to extract structure, text, and tables."""
    print(f"📄 Analyzing {os.path.basename(file_path)} with Azure Document Intelligence...")
    with open(file_path, "rb") as f:
        poller = doc_intel_client.begin_analyze_document("prebuilt-layout", document=f)
    result = poller.result()
    # result.content contains a highly accurate, reading-order representation of the document, including tables
    return result.content

def translate_to_english(text):
    """Translates text to English using Azure AI Translator."""
    print("🌍 Translating document to English...")
    path = '/translate?api-version=3.0&to=en'
    url = TRANSLATOR_ENDPOINT + path
    headers = {
        'Ocp-Apim-Subscription-Key': TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': TRANSLATOR_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    # Azure Translator has a 10k char limit. Our 1-page invoices are safe, but we slice just in case.
    body = [{'text': text[:9999]}] 
    request = requests.post(url, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

def chunk_and_embed(text):
    """Chunks the text and generates OpenAI embeddings."""
    print("🧠 Chunking and generating vectors...")
    # Simple chunking for 1-page invoices (splitting by double newline usually separates tables/paragraphs)
    chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 50]
    
    embedded_chunks = []
    for chunk in chunks:
        res = OPENAI_CLIENT.embeddings.create(input=chunk, model="text-embedding-3-small")
        embedded_chunks.append({
            "text": chunk,
            "vector": res.data[0].embedding
        })
    return embedded_chunks

def main():
    # 1. Ensure the index exists and can hold our metadata
    create_index()
    
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        print(f"Error: Could not find the '{documents_dir}' folder.")
        return

    # 2. Loop through every PDF in the folder
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(documents_dir, filename)
            
            # Step A: Intelligent Extraction (handles tables flawlessly)
            raw_text = extract_text_and_tables(file_path)
            
            # Step B: Translation
            english_text = translate_to_english(raw_text)
            
            # Step C: Chunking & Embedding
            embedded_data = chunk_and_embed(english_text)
            
            # Step D: Upload to Azure Search WITH METADATA
            print(f"☁️ Uploading chunks from {filename} to Azure AI Search...")
            search_documents = []
            for item in embedded_data:
                search_documents.append({
                    "id": str(uuid.uuid4()),
                    "content": item["text"],
                    "source_file": filename, # This allows the AI to cite its sources!
                    "content_vector": item["vector"]
                })
            
            search_client.upload_documents(documents=search_documents)
            print(f"✅ Successfully processed {filename}!\n")
            
    print("🎉 All documents processed! The NexaCorp Enterprise Index is live.")

if __name__ == "__main__":
    main()