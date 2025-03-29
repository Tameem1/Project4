import os
import logging
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader

from utils.constants import CHROMA_SETTINGS, get_vectorstore_path 
from utils.embedding_utils import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ingest_single_txt_file(file_path: str) -> bool:
    """
    Ingest a single .txt file into the vector store.

    The embeddings will be stored in a default chroma_dbs directory.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        bool: True if ingestion was successful, otherwise False.
    """
    if not os.path.isfile(file_path) or not file_path.lower().endswith(".txt"):
        logging.error(f"Invalid file path or not a .txt file: {file_path}")
        return False

    loaded_docs = [] 

    try:
        # Instantiate the TextLoader for .txt files
        loader = TextLoader(file_path)
        docs = loader.load()
        if not docs:
            logging.warning(f"No documents returned by loader for '{file_path}'")
        else:
            loaded_docs.extend(docs)
            logging.info(f"Loaded {len(docs)} document(s) from {file_path}")

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}", exc_info=True)
        return False

    if not loaded_docs:
        logging.warning(f"No valid content found for ingestion in {file_path}")
        return False

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(loaded_docs)
    logging.info(f"Splitting into chunks produced {len(chunks)} total chunks.")

    # 3. Load embeddings
    embeddings = get_embeddings()
    logging.info("Embeddings initialized successfully.")

    # 4. Determine the vector store path (using a default name for directory ingestion)
    persist_dir = get_vectorstore_path()
    os.makedirs(persist_dir, exist_ok=True)

    # 5. Initialize Chroma vector store
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    logging.info(f"Chroma vector store created/loaded at {persist_dir}.")

    # 6. Add documents to the vector store
    db.add_documents(chunks)
    db.persist()
    logging.info(f"Ingestion complete for {file_path}. Vector store updated.")

    return True

def ingest_directory_of_txt_files(directory_path: str) -> None:
    """
    Ingest all .txt files within a given directory.

    Args:
        directory_path (str): The path to the directory containing .txt files.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return

    txt_files_found = False
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".txt"):
            txt_files_found = True
            file_path = os.path.join(directory_path, filename)
            print(f"Ingesting: {file_path}")
            if ingest_single_txt_file(file_path):
                print(f"Successfully ingested: {file_path}")
            else:
                print(f"Failed to ingest: {file_path}")

    if not txt_files_found:
        print(f"No .txt files found in the directory: {directory_path}")

# Example usage for handling a directory of .txt files
if __name__ == "__main__":
    directory_to_ingest = "banking_text"

    ingest_directory_of_txt_files(directory_to_ingest)

