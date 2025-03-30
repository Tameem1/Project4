import os
from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredHTMLLoader

CHROMA_BASE_DIRECTORY = "utils/chroma_dbs"
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/chroma_dbs"

# Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# LLM Model configuration (If needed, adjust paths/names)
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", 
    "intfloat/multilingual-e5-large"  # Default value
)

def get_vectorstore_path() -> str:
    """
    Construct the path for the vector store where embeddings are stored
    for a given customer and chatbot.
    """
    return os.path.join(CHROMA_BASE_DIRECTORY)
