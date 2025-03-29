__import__('pysqlite3')
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
from transformers import AutoTokenizer

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

def configure_sidebar():
    """Set up the Islamic banking interface"""
    with st.sidebar:
        st.title("مساعد الخدمات المصرفية الإسلامية")
        st.markdown("""
        نظام ذكي متخصص في الخدمات المصرفية الإسلامية
        """)
        add_vertical_space(3)
        st.write("2025")

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
                
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            
            # Configure retriever
            retriever = db_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    "lambda_mult": 0.5,
                    "score_threshold": 0.35
                }
            )
            
            # Initialize LLM
            llm = ChatFireworks(
                api_key="fw_3ZfGXeDhjJfUxVHUVRBDfMeU",
                model="accounts/fireworks/models/qwen2p5-coder-32b-instruct",
                temperature=0.7,
                max_tokens=1500,
                top_p=1.0
            )
            
            prompt, memory = initialize_conversation_chain()
            
            st.session_state.components = {
                "retriever": retriever,
                "db_store": db_store,
                "tokenizer": tokenizer,
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

# Application Interface
configure_sidebar()
st.header("المساعد البنكي الإسلامي 💰")
initialize_components()

# Query Processing
user_query = st.text_input("اكتب استفسارك المصرفي هنا", placeholder="ما هي شروط القرض الإسلامي؟")


if user_query:
    try:
        # Validate query language
        if not is_arabic(user_query):
            st.warning("الاستفسار يحتوي على نص غير عربي")
            raise ValueError("Non-Arabic query")
            
        # Tokenization diagnostics
        tokens = st.session_state.components["tokenizer"].tokenize(user_query)
        logging.info(f"Query tokens: {tokens}")
        
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
        st.subheader("الإجابة")
        st.write(response.get("result", "لا تتوفر إجابة حاليا"))
        
        # Display sources
        if direct_docs:
            st.subheader("المصادر")
            for doc in direct_docs[:3]:
                source = doc.metadata.get("source", "غير معروف")
                st.write(f"- {source}")
                
        # Retrieval diagnostics
        st.subheader("تفاصيل الاسترجاع")
        st.json({
            "original_query": user_query,
            "retrieved_sources": [doc.metadata.get("source") for doc in direct_docs],
            "retrieval_score": [doc.metadata.get("score", 0.0) for doc in direct_docs],
            "token_count": len(tokens)
        })
        
    except Exception as e:
        logging.error(f"Query failed: {str(e)}", exc_info=True)
        st.error("حدث خطأ في المعالجة. يرجى المحاولة لاحقًا.")
