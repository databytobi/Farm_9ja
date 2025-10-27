import streamlit as st
import unicodedata
import re
import time
import tempfile
import os

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory

# New imports for custom retriever interface
from langchain.schema import BaseRetriever

# ---------------------------
# Setup API Keys
# ---------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------
# Request Limiter
# ---------------------------
if "last_requests" not in st.session_state:
    st.session_state.last_requests = []

def allow_request(limit=3, per_seconds=60):
    now = time.time()
    st.session_state.last_requests = [
        t for t in st.session_state.last_requests if now - t < per_seconds
    ]
    if len(st.session_state.last_requests) < limit:
        st.session_state.last_requests.append(now)
        return True
    return False

# ---------------------------
# Text Helpers
# ---------------------------
def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_repetitions(text, max_repeat=2):
    pattern = r'(\b\w+\b(?: \b\w+\b){0,3})\s*(?:\1\s*){' + str(max_repeat) + ',}'
    return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

# ---------------------------
# Translation Model
# ---------------------------
model_name = "HelpMumHQ/AI-translator-eng-to-9ja"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
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
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ---------------------------
# LLM (Gemini)
# ---------------------------
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
    src_name = supported_langs.get(src_lang, src_lang)
    tgt_name = supported_langs.get(tgt_lang, tgt_lang)

    prompt_translate = f"""
Translate the following text from {src_name} to {tgt_name}.
- Only return the translated text.
- Do NOT explain, interpret, or answer the question.

Text:
{text}
"""

    response = llm([HumanMessage(content=prompt_translate)])
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()

def translate_with_fallback(text, src_lang, tgt_lang):
    normalized_text = normalize_text(text)
    try:
        translated = translator_fn(normalized_text, src_lang, tgt_lang)
        cleaned = clean_repetitions(translated)
        if len(cleaned.strip()) < 5:
            return llm_translate(normalized_text, src_lang, tgt_lang)
        return cleaned.strip()
    except Exception:
        return llm_translate(normalized_text, src_lang, tgt_lang)

# ---------------------------
# Empty retriever (used when there are no docs)
# ---------------------------
class EmptyRetriever(BaseRetriever):
    """A minimal retriever that returns no documents (prevents passing None)."""
    def get_relevant_documents(self, query):
        return []

    async def aget_relevant_documents(self, query):
        return []

# ---------------------------
# Build RAG Chain with memory
# ---------------------------
def build_rag_chain(pdf_file=None):
    # Save uploaded PDF temporarily (if provided) or use bundled default path
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            pdf_path = tmp.name
    else:
        pdf_path = "data/best_practices_for_treating_crop_disease1.pdf"

    # Load PDF into LangChain document objects (if possible)
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception:
        documents = []

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents) if documents else []
    texts = [doc.page_content for doc in docs if isinstance(doc.page_content, str) and doc.page_content.strip()]

    # Create embeddings and vectorstore only if we have texts
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = None
    try:
        if texts:
            vectorstore = FAISS.from_texts(texts, embedding_model)
    except Exception as e:
        # If FAISS build fails, log and fall back to EmptyRetriever
        st.warning(f"Vectorstore build failed: {e}")

    # Ensure retriever is never None
    if vectorstore:
        # use default retriever parameters; adjust search_kwargs if desired
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    else:
        retriever = EmptyRetriever()

    # Prompt template (kept from your original)
    prompt_template = """
You are a professional Agronomist, Tutor, and Farming Advisor helping smallholder farmers.

Roles:
- Agronomist: Provide accurate, practical agricultural advice backed by science and best practices.
- Tutor: If the question is about "why something happens", explain step by step with an example like a teacher.
- Advisor: Always encourage learning and suggest what the farmer can try next or explore further.

Guidelines:
- If the user’s question is unclear, ask clarifying questions before answering.
- If the question is about "tools, fertilizers, or crops", include best practices and common mistakes.
- Always explain in clear, simple, farmer-friendly language.
- If unsure, say "I do not know" and then use DuckDuckGo Web Search.

Context:
{context}

Question:
{question}

Answer as an agronomist + tutor:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # ConversationalRetrievalChain with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # IMPORTANT: pass combine_docs_chain_kwargs a dict containing prompt (PromptTemplate)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory
    )

    # DuckDuckGo agent for web fallback
    ddg_search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description="Web search for farming topics. Use this tool when the provided context "
                        "does not contain the answer. It searches the web for agricultural information, "
                        "research, and best practices. Always summarize search results in clear, simple, "
                        "and actionable advice for farmers."
        )
    ]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    return qa_chain, agent

# ---------------------------
# Answer Function
# ---------------------------
def answer_question(user_question, user_lang, qa_chain, agent):
    if user_lang != 'en':
        query_en = translate_with_fallback(user_question, user_lang, 'en')
    else:
        query_en = user_question

    # The CRC (qa_chain) expects a dict input with "question"
    try:
        result = qa_chain({"question": query_en})
        rag_answer = result.get('answer', '') if isinstance(result, dict) else ''
    except Exception as e:
        # If chain fails for any reason, fall back to agent
        st.warning(f"Retrieval chain error: {e}")
        rag_answer = ''

    # If no good answer, use agent fallback (web)
    if not rag_answer or len(rag_answer.strip()) < 10 or "i do not know" in rag_answer.lower():
        try:
            rag_answer = agent.run(query_en)
        except Exception as e:
            rag_answer = f"Sorry — I couldn't find an answer. (agent error: {e})"

    if user_lang != 'en':
        rag_answer = translate_with_fallback(rag_answer, 'en', user_lang)

    return rag_answer

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Farm_9ja Assistant", page_icon="🌾", layout="centered")
st.title("🌾 Farm_9ja Agricultural Assistant")
st.markdown("""
Welcome! Ask your farming questions in English, Yoruba, Igbo, or Hausa.  
Get practical, actionable advice for smallholder farmers.
""")

with st.expander("🔒 Admin: Upload your own PDF knowledge base"):
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        st.success("PDF uploaded! All answers will be retrieved from this document.")

# Build chain AFTER possible PDF upload (so upload is used)
qa_chain, agent = build_rag_chain(uploaded_pdf) if uploaded_pdf else build_rag_chain()

# Language dropdown
lang_choice = st.selectbox(
    "🌍 Choose your language:",
    options=["English", "Yoruba", "Igbo", "Hausa"],
    index=0
)

lang_map = {"English": "en", "Yoruba": "yo", "Igbo": "ig", "Hausa": "ha"}
selected_lang = lang_map[lang_choice]

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask your farming question..."):
    if not allow_request():
        st.warning("⚠ Too many requests. Please wait before asking again.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(user_input, selected_lang, qa_chain, agent)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


