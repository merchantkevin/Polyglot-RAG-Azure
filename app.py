import streamlit as st
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

st.set_page_config(page_title="Multilingual RAG Portfolio", page_icon="🌍")

# --- Architecture Sidebar ---
st.sidebar.title("🌍 Multilingual RAG")
st.sidebar.markdown("""
**Architecture Overview:**
* **Translation:** Azure AI Translator (F0)
* **Database:** Azure AI Search (F0 Vector Index)
* **Embeddings:** OpenAI text-embedding-3-small
* **LLM:** GPT-4o

*Built for the Azure AI-102 Certification Prep.*
""")

# --- Setup Secure Clients ---
# These are pulled directly from Streamlit's encrypted cloud vault
OPENAI_CLIENT = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
INDEX_NAME = "multilingual-portfolio-index"

@st.cache_resource
def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT, 
        index_name=INDEX_NAME, 
        credential=AzureKeyCredential(SEARCH_KEY)
    )

search_client = get_search_client()

# --- Chat Interface ---
st.title("Chat with my Multilingual Documents 📄")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Azure AI Search and generating answer..."):
            try:
                # 1. Embed Question
                embedding_response = OPENAI_CLIENT.embeddings.create(
                    input=prompt, model="text-embedding-3-small"
                )
                question_vector = embedding_response.data[0].embedding

                # 2. Search Azure F0
                vector_query = VectorizedQuery(
                    vector=question_vector, k_nearest_neighbors=3, fields="content_vector"
                )
                results = search_client.search(
                    search_text=None, vector_queries=[vector_query], select=["content"]
                )
                context = "\n\n".join([doc['content'] for doc in results])

                # 3. Ask GPT-4o
                system_prompt = f"You are a helpful enterprise assistant. Answer using ONLY the following retrieved context:\n{context}"
                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")