# 🌍 Azure Multilingual RAG Pipeline

**Built for Azure AI-102 Certification Prep | Cost to run: $0.00 (Strictly Free Tiers)**

## Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) pipeline that ingests foreign-language documents, normalizes them to English via Azure AI services, and allows users to query the index using an external LLM.

## 🏗️ Architecture & Cloud Infrastructure
* **Document Ingestion:** Local PDF chunking via Python.
* **Translation Engine:** Azure AI Translator (F0 Free Tier).
* **Vector Database:** Azure AI Search (F0 Free Tier Vector Index).
* **Embeddings:** OpenAI `text-embedding-3-small`.
* **LLM / Generation:** OpenAI `gpt-4o`.
* **User Interface:** Streamlit (Coming soon!).

## 🚀 How It Works
1. **Extraction:** Reads a foreign-language PDF (e.g., Spanish).
2. **Translation:** Chunks the text and translates it to English using the Azure Translator REST API.
3. **Vectorization:** Converts the translated text into 1536-dimensional embeddings.
4. **Indexing:** Pushes the embedded chunks to a custom Azure AI Search index.
5. **Retrieval:** Users ask questions, which are embedded and queried against the Azure F0 index using HNSW Vector Search. The context is passed to GPT-4o for the final answer.