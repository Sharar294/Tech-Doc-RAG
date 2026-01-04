from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

file_path = "./data/datasheet.pdf"
persist_path = "./chroma_langchain_db"

loader = PyPDFLoader(file_path)

doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunked_docs = text_splitter.split_documents(doc)

embeddings = OllamaEmbeddings(
    model="llama3",
)

vector_store = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory=persist_path,
)