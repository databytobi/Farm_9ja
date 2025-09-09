import streamlit as st
import unicodedata
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
import torch
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from fpdf import FPDF
import tempfile

import tempfile, os, requests

# Load environment variables
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



# Normalization Function
def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Remove repeated phrases
def clean_repetitions(text, max_repeat=2):
    pattern = r'(\b\w+\b(?: \b\w+\b){0,3})\s*(?:\1\s*){' + str(max_repeat) + ',}'
    cleaned_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    return cleaned_text


# Translation model
model_name = "HelpMumHQ/AI-translator-eng-to-9ja"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# Force CPU usage to avoid NotImplementedError on Streamlit Cloud
device = "cpu"
translator = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)


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


# LLM wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY
)

supported_langs = {
    'yo': 'yoruba',
    'ig': 'igbo',
    'ha': 'hausa',
    'en': 'english'
}

def llm_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text from source language to target language using the LLM.
    This function is ONLY for translation (no explanations, no answering).
    """

    src_name = supported_langs.get(src_lang, src_lang)
    tgt_name = supported_langs.get(tgt_lang, tgt_lang)

    prompt_translate = f"""
    Translate the following text from {src_name} to {tgt_name}.
    - Only return the translated text.
    - Do NOT explain, interpret, or answer the question.
    - Keep meaning accurate and natural.

    Text:
    {text}
    """

    response = llm([HumanMessage(content=prompt_translate)])
    return response.content.strip()



def translate_with_fallback(text, src_lang, tgt_lang):
    """
    Always try transformer model first (fast + specialized).
    If it fails or gives poor result, fallback to Gemini LLM.
    """
    normalized_text = normalize_text(text)

    try:
        # Primary: Transformer model
        translated = translator_fn(normalized_text, src_lang, tgt_lang)
        cleaned = clean_repetitions(translated)

        # If translation looks too short or suspicious, fallback
        if len(cleaned.strip()) < 5:
            fallback = llm_translate(normalized_text, src_lang, tgt_lang)
            return fallback.strip()
        else:
            return cleaned.strip()
    except Exception:
        # Fallback if transformer completely fails
        return llm_translate(normalized_text, src_lang, tgt_lang)


def build_rag_chain(pdf_file=None):
    if pdf_file:
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getbuffer())  # ✅ use getbuffer() instead of read()
            pdf_path = tmp.name
    else:
        # Default knowledge base PDF (make sure this exists in your repo!)
        pdf_path = "data/best_practices_for_treating_crop_disease1.pdf"

    # Load PDF into LangChain
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs if isinstance(doc.page_content, str) and doc.page_content.strip()]

    return texts  # (or however you continue to build embeddings/vectorstore/qa_chain)
    
    # Enhanced prompt template with one-shot example and professional instructions
    prompt_template = """
You are a professional agricultural assistant for smallholder farmers.

Your job is to answer farming questions using the provided context or, if the answer is not in the context, by using the DuckDuckGo Web Search tool.

Important rules:
- Always answer clearly, concisely, and in simple language.
- Provide actionable advice farmers can immediately apply.
- If unsure, say "I do not know" and then use DuckDuckGo Web Search.
- Do not attempt translation. Assume the input is already in English and the output will be translated back if needed.

Example:
Context:
Maize is best planted at the start of the rainy season. Use certified seeds for optimal yield.

Question:
When should I plant maize for best results?

Answer:
You should plant maize at the start of the rainy season and use certified seeds for the best yield.

Now answer the following:

Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff",
        return_source_documents=True, chain_type_kwargs={"prompt": prompt},
    )
    ddg_search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description="Web search for farming topics, Use this tool when the provided context does not contain the answer, It searches the web for agricultural information, research, and best practices, Always summarize search results in clear, simple, and actionable advice for farmers."
        )
    ]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    return qa_chain, agent

def answer_question(user_question, user_lang, qa_chain, agent):
    """
    Main QA flow:
    1. Translate user question -> English
    2. Run RAG QA
    3. If RAG weak -> use DuckDuckGo agent
    4. Translate answer back -> user language
    """
    user_lang = user_lang if user_lang in supported_langs else 'en'

    # Step 1: Translate user question to English
    if user_lang != 'en':
        query_en = translate_with_fallback(user_question, user_lang, 'en')
    else:
        query_en = user_question

    # Step 2: Run through RAG
    result = qa_chain.invoke({"query": query_en})
    rag_answer = result['result']

    # Step 3: Fallback to web search if RAG fails
    if len(rag_answer.strip()) < 10 or "i do not know" in rag_answer.lower():
        rag_answer = agent.run(query_en)

    # Step 4: Translate back to user’s language
    if user_lang != 'en':
        rag_answer = translate_with_fallback(rag_answer, 'en', user_lang)

    return rag_answer


#def get_font_path():
#    return os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")

#def generate_pdf(question, answer):
#    pdf = FPDF()
#    pdf.add_page()
#    font_path = get_font_path()
#    pdf.add_font("DejaVu", "", font_path, uni=True)
#    pdf.set_font("DejaVu", size=12)
#    pdf.cell(0, 10, "Farm_9ja Agricultural Assistant", ln=True, align="C")
#    pdf.ln(10)
#    pdf.multi_cell(0, 10, f"Question:\n{question}\n\nAnswer:\n{answer}")
#    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#    pdf.output(temp_file.name)
#    return temp_file.name


# Streamlit UI
st.set_page_config(page_title="Farm_9ja Assistant", page_icon="🌾", layout="centered")
st.title("🌾 Farm_9ja Agricultural Assistant")
st.markdown("""
Welcome! Ask your farming questions in English, Yoruba, Igbo, or Hausa.  
Get practical, actionable advice for smallholder farmers.

*Admin Upload:* You can upload your own PDF to use as the knowledge base for answering questions.
""")

with st.expander("🔒 Admin: Upload your own PDF knowledge base"):
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        st.success("PDF uploaded! All answers will be retrieved from this document.")

# Build RAG chain depending on whether a PDF was uploaded
if uploaded_pdf:
    qa_chain, agent = build_rag_chain(uploaded_pdf)
else:
    qa_chain, agent = build_rag_chain()

with st.form("question_form"):
    lang_code = st.selectbox(
        "Select your language",
        options=list(supported_langs.keys()),
        format_func=lambda x: supported_langs[x].capitalize()
    )
    question = st.text_area("Enter your farming question:", height=100)
    submitted = st.form_submit_button("Get Answer")

if submitted and question.strip():
    with st.spinner("Thinking..."):
        answer = answer_question(question, lang_code, qa_chain, agent)
    st.markdown("#### Answer:")
    st.success(answer)
#    pdf_file_path = generate_pdf(question, answer)
#    with open(pdf_file_path, "rb") as f:
#        st.download_button(
#            label="Download answer as PDF",
#            data=f.read(),
#            file_name="Farm_9ja_Answer.pdf",
#            mime="application/pdf"

#        )


