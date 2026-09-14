from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(pdf):
    """Load a PDF and split it into overlapping chunks for retrieval."""

    loader = PyPDFLoader(pdf)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    return text_splitter.split_documents(documents)