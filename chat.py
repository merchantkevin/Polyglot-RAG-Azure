import os
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from promptflow.client import PFClient

# Load the Azure Search keys from .env
load_dotenv()

# 1. Securely retrieve the OpenAI key from our local Prompt Flow connection
print("Authenticating with local Prompt Flow vault...")
pf = PFClient()
connection = pf.connections.get(name="my_openai_conn")
openai_api_key = connection.secrets["api_key"]

# 2. Set up our AI and Search Clients
OPENAI_CLIENT = OpenAI(api_key=openai_api_key)
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "multilingual-portfolio-index"

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT, 
    index_name=INDEX_NAME, 
    credential=AzureKeyCredential(SEARCH_KEY)
)

def chat_loop():
    print("\n" + "="*50)
    print("🤖 Multilingual RAG Chat Initialized")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("="*50)

    # The Interactive Loop
    while True:
        # Wait for user input
        user_question = input("\n🧑 You: ")
        
        # Check if the user wants to quit
        if user_question.lower() in ['exit', 'quit']:
            print("\nEnding chat. Goodbye! 👋\n")
            break
            
        # Ignore empty inputs
        if not user_question.strip():
            continue

        try:
            # Step A: Convert the user's question into a vector
            embedding_response = OPENAI_CLIENT.embeddings.create(
                input=user_question,
                model="text-embedding-3-small"
            )
            question_vector = embedding_response.data[0].embedding

            # Step B: Search the F0 Azure AI Search index for the closest matches
            vector_query = VectorizedQuery(
                vector=question_vector, 
                k_nearest_neighbors=3, 
                fields="content_vector"
            )
            
            results = search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["content"]
            )

            # Step C: Combine the retrieved chunks into a single context string
            context = "\n\n".join([doc['content'] for doc in results])
            
            # Step D: Pass the context and the question to GPT-4o
            system_prompt = f"""
            You are a helpful enterprise assistant. Answer the user's question using ONLY the following retrieved context.
            If the answer is not in the context, say "I cannot answer this based on the provided document."
            
            Context:
            {context}
            """
            
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ]
            )
            
            # Print the AI's answer
            print(f"\n🤖 AI: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

# --- Run the Chat ---
if __name__ == "__main__":
    chat_loop()