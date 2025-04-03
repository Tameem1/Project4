__import__('pysqlite3')
import os
import sys
import logging
import torch
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# AI/ML Components
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_fireworks import ChatFireworks
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate

# Configuration
from utils.constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY
)

# Arabic character range check
ARABIC_CHAR_RANGE = (0x0600, 0x06FF)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("retrieval_diagnostics.log"), logging.StreamHandler()]
)

def is_arabic(text: str, threshold=0.6) -> bool:
    """Check if text is primarily Arabic"""
    arabic_chars = sum(1 for c in text if ord(c) >= ARABIC_CHAR_RANGE[0] and ord(c) <= ARABIC_CHAR_RANGE[1])
    return (arabic_chars / max(len(text), 1)) > threshold

def initialize_conversation_chain():
    """Arabic conversation template"""
    conversation_template = """استخدم السياق التالي للإجابة على السؤال في النهاية.
    إذا لم تعرف الإجابة، قل أنك لا تعرف فقط.
    
    السياق: {context}
    
    التاريخ: {history}
    السؤال: {question}
    الإجابة المفيدة:"""
    
    return PromptTemplate(
        input_variables=["history", "context", "question"],
        template=conversation_template
    ), ConversationBufferMemory(
        input_key="question",
        memory_key="history"
    )


def detect_computation_device():
    """Hardware detection"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def initialize_components():
    """Initialization without text normalization"""
    if "components" not in st.session_state:
        try:
            device = detect_computation_device()
            
            # Initialize embeddings
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device}
            )
            
            # Initialize Chroma
            db_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS,
            )
            
            # Collection verification
            collection = db_store._client.get_collection(name=db_store._collection.name)
            doc_count = collection.count()
            logging.info(f"Chroma DB documents: {doc_count}")
            
            if doc_count == 0:
                raise ValueError("Database is empty")
            
            # Configure retriever
            retriever = db_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    "lambda_mult": 0.5,
                    "score_threshold": 0.35
                }
            )
            
            # Get API key from environment
            fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
            if not fireworks_api_key:
                st.error("Missing FIREWORKS_API_KEY in environment variables")
                raise ValueError("FIREWORKS_API_KEY not set")
            
            # Get model name from environment with fallback
            model_name = os.environ.get(
                "FIREWORKS_MODEL_NAME", 
                "accounts/fireworks/models/qwen2p5-coder-32b-instruct"
            )

            # Initialize LLM with environment variables
            llm = ChatFireworks(
                api_key=fireworks_api_key,  # From environment
                model=model_name,           # From environment
                temperature=0.7,
                max_tokens=1500,
                top_p=1.0
            )
            
            prompt, memory = initialize_conversation_chain()
            
            st.session_state.components = {
                "retriever": retriever,
                "db_store": db_store,
                "llm": llm,
                "chain": RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": prompt,
                        "memory": memory
                    }
                )
            }
            
        except Exception as e:
            logging.error("Initialization failed", exc_info=True)
            st.error("فشل تحميل النظام. يرجى التحقق من السجلات.")
            raise



# Text direction CSS
st.markdown("""
    <style>
        .stTextInput input {
            direction: rtl;
            text-align: right;
            padding-right: 20px;
        }
        .rtl-text {
            direction: rtl;
            text-align: right;
            unicode-bidi: embed;
        }
        /* New header/footer removal */
         #stDeployButton {display:none;}
         footer {visibility: hidden;}
         [data-testid="stHeader"] {display: none;}
         [data-testid="stToolbar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Main content
st.markdown("<div class='rtl-text'><h1>المساعد البنكي الإسلامي 💰</h1></div>", unsafe_allow_html=True)
initialize_components()

# Query Processing
user_query = st.text_input(
    " ",
    placeholder="ما هي شروط القرض الإسلامي؟",
    label_visibility="collapsed"
)

if user_query:
    try:
        # Validate query language
        if not is_arabic(user_query):
            st.warning("الاستفسار يحتوي على نص غير عربي")
            raise ValueError("Non-Arabic query")
        with st.spinner("جاري معالجة سؤالك، يرجى الانتظار..."):
            # Retrieve documents
            direct_docs = st.session_state.components["retriever"].get_relevant_documents(user_query)
            
            # Fallback mechanism
            if not direct_docs:
                logging.warning("Primary retrieval failed - using fallback")
                direct_docs = st.session_state.components["db_store"].max_marginal_relevance_search(
                    user_query,
                    k=5,
                    fetch_k=20
                )
                
            # Process response
            response = st.session_state.components["chain"]({"query": user_query})
        
        # Display results
        st.markdown("<div class='rtl-text'><h2>الإجابة</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rtl-text'>{response.get('result', 'لا تتوفر إجابة حاليا')}</div>", 
                    unsafe_allow_html=True)
        
        # Display sources
        if direct_docs:
            st.markdown("<div class='rtl-text'><h2>المصادر</h2></div>", unsafe_allow_html=True)
            for doc in direct_docs[:3]:
                source = doc.metadata.get("source", "غير معروف")
                st.markdown(f"<div class='rtl-text'>- {source}</div>", 
                            unsafe_allow_html=True)
                
    except Exception as e:
        logging.error(f"Query failed: {str(e)}", exc_info=True)
        st.error("حدث خطأ في المعالجة. يرجى المحاولة لاحقًا.")
