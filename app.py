import streamlit as st
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

st.set_page_config(page_title="NexaCorp Global AI Portal", page_icon="🏢")

# --- Architecture Sidebar ---
st.sidebar.title("🏢 NexaCorp AI Portal")
st.sidebar.markdown("""
**Enterprise RAG Architecture:**
* **Ingestion:** Azure AI Document Intelligence (Prebuilt Layout)
* **Translation:** Azure AI Translator
* **Database:** Azure AI Search
* **Embeddings:** OpenAI text-embedding-3-small
* **LLM:** GPT-4o

*Built for Azure AI-102 Certification Prep.*
""")

# --- Setup Secure Clients ---
OPENAI_CLIENT = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
INDEX_NAME = "nexacorp-global-index" # Pointing to our new enterprise index!

@st.cache_resource
def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT, 
        index_name=INDEX_NAME, 
        credential=AzureKeyCredential(SEARCH_KEY)
    )

search_client = get_search_client()

# --- User Onboarding & Context ---
st.title("Global Procurement & Finance Portal")

st.info("""
**👋 Welcome to the NexaCorp Auditor Demo!**

I have ingested the global vendor documents for NexaCorp, A fake multinational company with offices in Europe and USA. This includes documents in various languages (Spanish Hardware Invoices, French Logistics POs, German Real Estate Leases, etc.). 
My Azure pipeline used **Document Intelligence** to extract the tables, translated them to English using the Azure AI Translator, and indexed them.

**Try asking auditing questions for NexaCorp's global spend:**
* *"What is our total monthly base rent in Munich?"*
* *"How many laptops did the Madrid office buy, and what was the IVA tax applied?"*
* *"What is the fuel surcharge for the Paris freight shipments?"*
""")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about our global vendors or contracts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching global vendor documents..."):
            try:
                # 1. Embed Question
                embedding_response = OPENAI_CLIENT.embeddings.create(
                    input=prompt, model="text-embedding-3-small"
                )
                question_vector = embedding_response.data[0].embedding

                # 2. Search Azure (Now retrieving the source_file metadata too!)
                vector_query = VectorizedQuery(
                    vector=question_vector, k_nearest_neighbors=3, fields="content_vector"
                )
                results = search_client.search(
                    search_text=None, vector_queries=[vector_query], select=["content", "source_file"]
                )
                
                # Format the context so the LLM knows WHICH file each chunk came from
                context_blocks = []
                for doc in results:
                    context_blocks.append(f"[Source: {doc['source_file']}]\n{doc['content']}")
                context = "\n\n".join(context_blocks)

                # 3. Ask GPT-4o (Now with Conversational Memory!)
                system_prompt = f"""You are a smart enterprise financial and legal auditing assistant for NexaCorp. 
                
                <context>
                {context}
                </context>

                Rules:
                1. Answer the user's question using ONLY the <context> block AND our previous conversation history.
                2. If the user asks a follow-up math question (like "add them up" or "what is the difference"), use the exact data and numbers from your previous answers in the chat history to calculate it.
                3. You MUST cite the source file you used. If calculating based on previous answers, cite the sources from those previous answers.
                4. If the context and history do not contain the answer, reply EXACTLY with: "I cannot find this information currently, please reframe the question!"
                """
                
                # Build the memory payload for OpenAI
                messages_payload = [{"role": "system", "content": system_prompt}]
                
                # Appending the recent chat history (last 6 messages to save on token costs)                
                for msg in st.session_state.messages[-6:]: 
                    messages_payload.append({"role": msg["role"], "content": msg["content"]})
                
                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    messages=messages_payload
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")