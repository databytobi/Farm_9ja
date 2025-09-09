import unicodedata
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
import google.generativeai as genai
import torch
from dotenv import load_dotenv
import os
import PyPDF2
import sentencepiece as spm
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("Google Key:", google_api_key[:5] + "*")  # just to confirm it's loaded

# Normalization Function: preserves diacritics, cleans whitespace
def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Post-processing: remove repeated phrases > max_repeat times
def clean_repetitions(text, max_repeat=2):
    pattern = r'(\b\w+\b(?: \b\w+\b){0,3})\s*(?:\1\s*){' + str(max_repeat) + ',}'
    cleaned_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    return cleaned_text

# Load HelpMumHQ translation model & tokenizer
model_name = "HelpMumHQ/AI-translator-eng-to-9ja"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translator = M2M100ForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Translator function call using HelpMumHQ model
def translator_fn(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(translator.device)
    generated_tokens = translator.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_new_tokens=512
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

# Define Gemini LLM wrapper
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

# Supported language codes
supported_langs = {
    'yo': 'yoruba',
    'ig': 'igbo',
    'ha': 'hausa',
    'en': 'english'
}

# LLM translator function using prompt-based translation
def llm_translate(text, src_lang, tgt_lang):
    prompt_translate = f"Translate this text from {supported_langs.get(src_lang, src_lang)} to {supported_langs.get(tgt_lang, tgt_lang)}:\n\n{text}"
    response = llm([HumanMessage(content=prompt_translate)])
    return response.content

# Translation with fallback to LLM if output is missing or repetitive
def translate_with_fallback(text, src_lang, tgt_lang):
    normalized_text = normalize_text(text)
    try:
        translated = translator_fn(normalized_text, src_lang, tgt_lang)
        cleaned = clean_repetitions(translated)
        if len(cleaned.strip()) < 10 or cleaned == translated:
            fallback = llm_translate(normalized_text, src_lang, tgt_lang)
            return fallback
        else:
            return cleaned
    except Exception as e:
        return llm_translate(normalized_text, src_lang, tgt_lang)

# Step 1: Load PDF
pdf_path = "data/best_practices_for_treating_crop_disease1.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Step 2: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3: Extract only the text content
texts = [doc.page_content for doc in docs]
texts = [t for t in texts if isinstance(t, str) and t.strip()]

# Step 4: Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create embeddings for the texts
embeddings = embedding_model.embed_documents(texts)
vectorstore = FAISS.from_texts(texts, embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

print(f"✅ Created {len(embeddings)} embeddings from {len(texts)} chunks")

# Prompt template for RAG answering
prompt_template = """
You are a friendly expert agricultural assistant for smallholder farmers.

Use the following context to answer the question comprehensively, clearly, and with practical advice.

Context:

{context}

Question:
{question}

Instructions:
- Use simple, non-technical language.
- Provide actionable tips farmers can immediately apply.
- Explain terms if complex vocabulary is necessary.
- If unsure, honestly say you do not know.

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff",
    return_source_documents=True, chain_type_kwargs={"prompt": prompt},
)

# DuckDuckGo search fallback tool
ddg_search = DuckDuckGoSearchRun()
tools = [Tool(name="DuckDuckGo Search", func=ddg_search.run, description="Web search for farming topics")]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Final function answering questions with translation, RAG, and fallback logic
def answer_question(user_question, user_lang):
    user_lang = user_lang if user_lang in supported_langs else 'en'
    if user_lang != 'en':
        query_en = translate_with_fallback(user_question, user_lang, 'en')
    else:
        query_en = user_question
    result = qa_chain.invoke({"query": query_en})
    rag_answer = result['result']
    if len(rag_answer.strip()) < 10 or "I don't know" in rag_answer.lower():
        rag_answer = agent.run(query_en)
    if user_lang != 'en':
        rag_answer = translate_with_fallback(rag_answer, 'en', user_lang)
    return rag_answer

if __name__ == "__main__":
    print("Supported languages: yo (Yoruba), ig (Igbo), ha (Hausa), en (English)")
    user_lang_code = input("Enter language code: ").strip().lower()
    user_question = input("Enter your question: ").strip()
    final_answer = answer_question(user_question, user_lang_code)
    print("\nAnswer:\n", final_answer)