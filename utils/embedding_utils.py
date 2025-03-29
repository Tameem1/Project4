from langchain.embeddings import HuggingFaceEmbeddings
from .constants import EMBEDDING_MODEL_NAME

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}  # Adjust device if needed
    )