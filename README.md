# 🏢 Polyglot RAG using Azure 

**Live Demo:** https://polyglot-rag-azure-ehaqvxwdrtbt4mbmsvc2mt.streamlit.app/

**Built for Azure AI-102 Certification Prep**

## 📊 The Business Problem
Multinational corporations deal with highly structured, multilingual vendor documents (invoices, purchase orders, leases). Traditional search and basic OCR fail to capture table structures or bridge the language gap, forcing auditors into manual, time consuming data extraction.

## 🚀 The Solution
This project is an end-to-end Retrieval Augmented Generation (RAG) pipeline built for a synthetic Fortune 500 company (NexaCorp). It ingests foreign language vendor PDFs, flawlessly extracts complex tables, translates the data to English, and provides a conversational interface that **cites its exact sources**.

## 🏗️ Cloud Infrastructure & Architecture
* **Intelligent Extraction:** Azure AI Document Intelligence (`prebuilt-layout`) extracts text and complex table structures from PDFs.
* **Translation Engine:** Azure AI Translator normalizes foreign vendor data (Spanish, French, German, etc.) into English.
* **Vector Database:** Azure AI Search (Vector Index) stores embeddings alongside metadata (`source_file`).
* **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions).
* **LLM / Generation:** OpenAI `gpt-4o` for enterprise grounding.
* **User Interface:** Streamlit Community Cloud.

## ✨ Key Features
1. **Multi Document Indexing:** Processes an entire directory of disparate vendor documents into a single unified search index.
2. **Table Preservation:** Overcomes standard `pypdf` limitations by using Document Intelligence to maintain line-item tax and pricing tables.
3. **Cross Lingual Querying:** Users can ask questions in English about a Spanish invoice or a German lease and get mathematically accurate answers.
4. **Source Citation:** The LLM is strictly prompted to append the exact `source_file` metadata to every answer to ensure auditability and trust.

## 🛠️ How to Run Locally
1. Clone the repository: `git clone https://github.com/merchantkevin/Polyglot-RAG-Azure.git`
2. Create a virtual environment and run `pip install -r requirements.txt`.
3. Create a `.env` file with your Azure (Search, Translator, Document Intelligence) and OpenAI API keys.
4. Place sample PDFs in a `documents/` folder.
5. Run the ingestion pipeline: `python pipeline.py`
6. Launch the UI: `streamlit run app.py`

## 💡 Future Enhancements
* Semantic Caching: Storing commonly asked questions and respective answers in a fast cache to reply faster while saving LLM tokens.
* Automated Ingestion: When admins add a file to the storage, they won't need to run the pipeline manually, instead the pipeline will run automatically upon file upload
* Source file access: Users should be able to find the exact space AI found it's answer from. Access should be based on company policy.

Suggest some more enhancements!
