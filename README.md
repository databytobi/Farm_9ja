
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






🌱 Farm 9ja – Multilingual Crop Advisory RAG App

Farm 9ja is an AI-powered multilingual farming assistant that helps farmers access reliable agricultural knowledge in English, Yoruba, Igbo, and Hausa.
The app uses advanced AI technologies including translation models, Retrieval-Augmented Generation (RAG), and web search fallback to deliver accurate and helpful responses and a conversational agent to provide accurate, localized answers about crop diseases, treatments, and best farming practices.


## 🚀 Project Overview

Farm_9ja enables farmers to ask questions in their native language and receive expert agricultural advice. It supports uploading custom PDF documents to serve as the knowledge base and uses a combination of transformer models and Gemini LLM to ensure high-quality translations and answers.

🚀 Features

📖 Knowledge Base Integration: Load agricultural best-practice PDFs (default provided).

🔎 RAG (Retrieval-Augmented Generation): Combines embeddings + vector search + LLM for context-aware answers.

🌍 Multilingual Support: Ask questions in English, Yoruba, Igbo, or Hausa.

💬 Interactive Chat: Farmers can query in their preferred language.

🧠 *Gemini LLM*: Provides fallback answers when context is insufficient

🌐 *DuckDuckGo Web Search*: Searches the web for farming information

🔄 *Translation fallback*: Uses LLM if transformer translation fails

🧹 *Text normalization and repetition cleaning* for better quality

📂 Custom Uploads: Upload your own PDFs for tailored advisory.

🛡 Fallback System: If PDFs are missing/unreadable, the app runs gracefully with limited features.


🛠 Tech Stack

Frontend: Streamlit

Transformers (M2M100) – Translation model

Google Gemini LLM – Language model for fallback translation and QA

DuckDuckGo Search – Web search tool

LLM & RAG: LangChain, HuggingFace Embeddings, FAISS

PDF Loader: PyPDFLoader

Languages: English, Yoruba, Igbo, Hausa

Deployment: Compatible with Streamlit Cloud / Docker




📂 Project Structure

farm_9ja/
│
├── app.py                        # Main Streamlit app  
├── data/  
│   └── best_practices_for_treating_crop_disease1.pdf   # Default knowledge base  
├── requirements.txt              # Python dependencies  
├── .gitignore                    # Git ignore file  
└── README.md                     # Project documentation


⚡ Installation

1. Clone the repository:



git clone https://github.com/your-username/farm_9ja.git
cd farm_9ja

2. Create & activate a virtual environment:



python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3. Install dependencies:



pip install -r requirements.txt


▶ Running the App

streamlit run app.py

The app will open in your browser at http://localhost:8501/.



🌍 Usage

Select your language (English, Yoruba, Igbo, Hausa).

Ask a question (e.g., “How often should I water tomatoes?”).

The app retrieves knowledge from the PDF and provides an answer.

You can also upload your own PDF knowledge base.


📝 Example Questions

English: When is the best time to plant maize?

Yoruba: Nigbawo ni akoko to dara jùlọ lati gbin agbado?

Igbo: Kedu oge kacha mma iji kụọ oka?

Hausa: Yaushe ne mafi dacewa a dasa masara?


🔮 Future Improvements

Add speech-to-text for voice-based farmer queries.

Extend to more local Nigerian languages.

Connect to real-time weather & market APIs.

Mobile-friendly version for offline usage.


🤝 Contributing

Contributions are welcome!

1. Fork the repo


2. Create a feature branch (git checkout -b feature-name)


3. Commit changes (git commit -m "Added feature")


4. Push to branch (git push origin feature-name)


5. Open a Pull Request



📜 License

This project is licensed under the MIT License – feel free to use, modify, and share.

