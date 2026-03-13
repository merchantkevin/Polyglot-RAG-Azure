# 🌍 Polyglot-RAG-Azure

**Live Demo:** [Link to your Streamlit App will go here]

**Built for Azure AI-102 Certification Prep**

## Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) web application. It ingests foreign-language documents, normalizes them to English via Azure AI services, and allows users to query the index through a sleek UI using an external LLM.

## 🏗️ Architecture & Cloud Infrastructure
* **Document Ingestion:** Local PDF chunking via Python.
* **Translation Engine:** Azure AI Translator (F0 Free Tier).
* **Vector Database:** Azure AI Search (F0 Free Tier Vector Index).
* **Embeddings:** OpenAI API `text-embedding-3-small`.
* **LLM / Generation:** OpenAI API `gpt-4o`.
* **User Interface:** Streamlit Community Cloud.

## 🚀 How It Works
1. **Extraction:** Reads a foreign-language PDF.
2. **Translation:** Translates text to English using the Azure Translator REST API.
3. **Vectorization:** Converts translated text into 1536-dimensional embeddings.
4. **Indexing:** Pushes embedded chunks to an Azure AI Search index.
5. **Retrieval & Chat:** Users ask questions via the Streamlit UI. Questions are embedded and queried against the Azure F0 index. The context is passed to GPT-4o for the final answer.

## 🛠️ How to Run Locally
1. Clone the repository and navigate to the folder.
2. Create a virtual environment and run `pip install -r requirements.txt`.
3. Create a `.env` file with your Azure and OpenAI API keys.
4. Run the backend pipeline: `python pipeline.py`
5. Launch the UI: `streamlit run app.py`