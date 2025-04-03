from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import shutil


load_dotenv()


CHROMA_PATH = "chroma"
DATA_PATH = "./data/"


def main():
    generate_data_store()
    # load_documents("law.pdf")

def save_to_txt(filename, text):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text)
    print(f"Text has been saved to {filename}")


def generate_data_store():
    paths  = os.listdir(DATA_PATH)
    for path in paths:
        documents = load_documents(path=path)
        for document in documents:
            document_string = document.page_content
            filename = f"convert_to_pdf/{path}_convert.txt"
        save_to_txt(filename, document_string)


def load_documents(path):
    target_path = DATA_PATH + path
    print(f"Loading documents from: {target_path}")
    loader = PyMuPDFLoader(file_path=target_path)
    documents = loader.load()
    for i in  range(len(documents)):
        documents[i] = Document(documents[i].page_content.replace("สํานักงานคณะกรรมการกฤษฎีกา", ""))
        documents[i] = Document(documents[i].page_content.replace("ส ำนักงำนคณะกรรมกำรกฤษฎีกำ", ""))
    return documents

if __name__ == "__main__":
    main()
