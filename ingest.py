import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

def load_file_to_documents(file_path: str) -> list[Document]:
    """
    Loads document content based on its file extension.
    Supports PDF, Word, CSV, Excel, and Text/Markdown.
    """
    ext = os.path.splitext(file_path)[1].lower()
    source_name = os.path.basename(file_path)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()
    elif ext == ".csv":
        loader = CSVLoader(file_path)
        return loader.load()
    elif ext in [".xlsx", ".xls"]:
        import pandas as pd
        xls = pd.ExcelFile(file_path)
        docs = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheet_text = f"Document: {source_name}\nSheet: {sheet_name}\n\n"
            for idx, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                sheet_text += f"Row {idx + 1}: {row_str}\n"
            
            docs.append(Document(
                page_content=sheet_text,
                metadata={"source": source_name, "sheet": sheet_name}
            ))
        return docs
    else:
        # Fallback for txt, md, log, config, etc.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": source_name})]
        except Exception as e:
            raise ValueError(f"Could not read text file {source_name}: {str(e)}")

def ingest_documents(file_paths: list[str], namespace: str) -> None:
    """
    Reads multiple files, splits their content into chunks, 
    generates embeddings, and uploads them to Pinecone under a specific namespace.
    """
    if not file_paths:
        return

    # ---------- Connect Pinecone ----------
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # ---------- Load Documents ----------
    all_documents = []
    for path in file_paths:
        print(f"Loading {os.path.basename(path)}...")
        try:
            docs = load_file_to_documents(path)
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {os.path.basename(path)}: {str(e)}")

    if not all_documents:
        raise ValueError("No text content could be extracted from the uploaded files.")

    # ---------- Split text ----------
    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(all_documents)

    # ---------- Embedding model ----------
    print("Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [doc.page_content for doc in splits]
    embeddings = embedding_model.embed_documents(texts)

    # ---------- Prepare vectors ----------
    print("Preparing vectors...")
    vectors = []
    for i, (doc, emb) in enumerate(zip(splits, embeddings)):
        metadata = {
            "text": doc.page_content,
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown")
        }
        # Include sheet if present for excel
        if "sheet" in doc.metadata:
            metadata["sheet"] = doc.metadata["sheet"]

        vectors.append({
            "id": f"chunk-{i}",
            "values": emb,
            "metadata": metadata
        })

    # ---------- Upload ----------
    print(f"Uploading vectors to Pinecone namespace '{namespace}'...")
    index.upsert(vectors=vectors, namespace=namespace)
    print("SUCCESS: Knowledge base updated successfully!")

if __name__ == "__main__":
    # Test script if executed directly
    print("Please run this script inside the streamlit app dashboard.")