
# 📘 Project Documentation: Farm_9ja Agricultural Assistant

## 1. Overview

Farm_9ja is a multilingual agricultural assistant designed to support smallholder farmers in Nigeria. Built using Streamlit and powered by advanced AI technologies, the application provides practical and actionable farming advice in *English, **Yoruba, **Igbo, and **Hausa*. It leverages Retrieval-Augmented Generation (RAG), transformer-based translation models, and fallback mechanisms using Google Gemini LLM and DuckDuckGo web search to ensure accurate and helpful responses.


## 2. Motivation

Smallholder farmers in Nigeria often face challenges accessing reliable agricultural information in their native languages. Language barriers, limited internet access, and lack of localized expertise hinder their productivity and decision-making. Farm_9ja was developed to bridge this gap by offering a user-friendly, multilingual platform that delivers expert farming advice tailored to local needs.

---

## 3. Key Components

- *Streamlit UI*: Provides an intuitive interface for users to interact with the assistant.
- *Translation Models*: Uses HelpMumHQ/AI-translator-eng-to-9ja for fast and specialized translation between English and Nigerian languages.
- *Google Gemini LLM*: Offers fallback translation and question answering when transformer models are insufficient.
- *PDF Knowledge Base*: Allows administrators to upload custom agricultural documents to serve as the source of truth.
- *LangChain RAG Pipeline*: Retrieves relevant information from the uploaded documents using FAISS vector store and HuggingFace embeddings.
- *DuckDuckGo Web Search Agent*: Performs web searches when the document context does not contain the answer.
- *Fallback System: Ensures the app still runs if no PDF is available.

## 4. Method

The application follows a structured pipeline to deliver accurate responses:
The workflow of the Farm 9ja system is as follows:

1. PDF Loading

Default agricultural best practices PDF is loaded, or admin uploads a new document.



2. Document Chunking & Embeddings

Text is split into small overlapping chunks (500 tokens, 50 overlap).

Embeddings are generated using HuggingFace’s all-MiniLM-L6-v2 model.



3. Vector Store Creation

Embeddings are stored in FAISS (Facebook AI Similarity Search) for efficient retrieval.



## 5. Benefits

- *Multilingual Accessibility*: Supports major Nigerian languages to ensure inclusivity.
- *Custom Knowledge Base*: Empowers admins to tailor the assistant to specific agricultural domains.
- *Reliable Answers*: Combines document retrieval and web search to maximize accuracy.
- *User-Friendly Interface*: Streamlit-based UI ensures ease of use for farmers with minimal technical skills.
- *Scalable Architecture*: Modular design allows easy integration of new models and tools.
- *Farmer-Centered: Designed to address real needs of smallholder farmers.

## 6. Conclusion

Farm_9ja is a powerful tool for democratizing agricultural knowledge among smallholder farmers in Nigeria. By combining multilingual support, AI-driven translation, and intelligent retrieval mechanisms, it delivers actionable farming advice that can significantly improve productivity and decision-making. The project stands as a testament to how technology can be harnessed to solve real-world problems in underserved communities.

---

Would you like me to generate a PDF version of this documentation or help you organize it into a GitHub wiki?
