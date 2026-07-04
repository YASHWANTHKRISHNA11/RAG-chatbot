import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from config import PINECONE_API_KEY, PINECONE_INDEX, GROQ_API_KEY

# Cache the embedding model at the module level to avoid loading it on every call
_embedding_model = None

def get_embeddings():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model

def get_rag_chain(namespace: str, model_name: str):
    """
    Returns a dynamic LangChain RetrievalQA chain pointing to the specified Pinecone namespace
    and configured with the chosen Groq model.
    """
    embedding = get_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=namespace
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=0
    )

    template = """
You are an AI assistant that answers questions strictly using the document.

Rules:
1. Only use the provided context
2. Do NOT use outside knowledge
3. If the answer is missing say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer clearly and briefly.
"""
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def ask_question(query: str, namespace: str, model_name: str) -> str:
    """
    Queries the dynamic QA chain and formats the output showing source documents and page numbers.
    """
    qa_chain = get_rag_chain(namespace, model_name)
    result = qa_chain.invoke({"query": query})

    answer = result["result"]
    docs = result["source_documents"]

    # Extract source names and page numbers
    sources = set()
    pages = set()
    for doc in docs:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])
        if "page" in doc.metadata and doc.metadata["page"] != "unknown":
            try:
                # Add 1 as pages are 0-indexed in PDF parsing
                pages.add(int(doc.metadata["page"]) + 1)
            except (ValueError, TypeError):
                pass

    ref_details = []
    if sources:
        ref_details.append(f"📄 Sources: {', '.join(sources)}")
    if pages:
        ref_details.append(f"📖 Page(s): {', '.join(map(str, sorted(pages)))}")

    if ref_details:
        answer += "\n\n" + " | ".join(ref_details)

    return answer