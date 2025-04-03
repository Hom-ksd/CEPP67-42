from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import json


load_dotenv()


CHROMA_PATH = "chroma"
DATA_PATH = "./convert_to_pdf/"

def main():
    generate_data_store()

def generate_data_store():
    paths  = os.listdir(DATA_PATH)
    mapping_chunk = {}
    chunk_index = 1
    for path in paths:
        documents = load_documents(path=path)
        chunks, number = split_text(documents, chunk_index)
        print(chunks[-1].metadata)
        save_chunks(chunks,chunk_index)
        mapping_chunk[f"{path}"] = (chunk_index, chunk_index + number - 2)
        print(f"mapping_chunk : \n{mapping_chunk}")
        chunk_index += number - 1
        save_to_chroma(chunks) 
        
    with open("mapping_chunk.json", "w") as json_file:
        json.dump(mapping_chunk, json_file, indent=4)


def load_documents(path):
    target_path = DATA_PATH + path
    print(f"Loading documents from: {target_path}")
    loader = PyMuPDFLoader(file_path=target_path)
    documents = loader.load()
    for i in  range(len(documents)):
        documents[i] = Document(documents[i].page_content.replace("สํานักงานคณะกรรมการกฤษฎีกา", ""))
        documents[i] = Document(documents[i].page_content.replace("ส ำนักงำนคณะกรรมกำรกฤษฎีกำ", ""))
    return documents


def split_text(documents: list[Document], start_chunk_index):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    chunk_counter = 1

    for i, chunk in enumerate(chunks, start=1):  # Start enumeration from 1
        chunk.metadata["chunk_number"] = chunk_counter
        chunk_counter += 1

    print(chunk_counter)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks, chunk_counter

def save_chunks(chunks,chunk_index):
    output_dir = "chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Save chunks to files

    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join(output_dir, f"document_chunk_{chunk_index + i}.txt")
        with open(chunk_filename, "w", encoding="utf-8") as file:
            file.write(chunk.page_content)

    print(f"Saved chunks for {len(chunks)} documents.")

def save_to_chroma(chunks: list[Document], batch_size=500, file_name="unknown_file"):
    # Load existing Chroma DB if it exists, otherwise create a new one
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True},  # Required for cosine similarity
    )

    # Ensure metadata includes chunk number and file name
    for chunk in chunks:
        if "chunk_number" not in chunk.metadata:
            chunk.metadata["chunk_number"] = "Unknown"  # Fallback value
        chunk.metadata["file_name"] = file_name  # Add file name to metadata

    # Extract texts and metadata separately
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Load or create Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model) if os.path.exists(CHROMA_PATH) else None

    # Process chunks in batches to avoid exceeding memory limits
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        if db:
            db.add_texts(batch_texts, metadatas=batch_metadatas)  # Append new documents
        else:
            db = Chroma.from_texts(
                batch_texts, embedding_model, metadatas=batch_metadatas, persist_directory=CHROMA_PATH
            )

        print(f"Saved batch {i // batch_size + 1} of {len(texts) // batch_size + 1} to Chroma.")

    db.persist()
 



if __name__ == "__main__":
    main()
