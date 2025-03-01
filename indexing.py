import re
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders import PyMuPDFLoader
import pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

directory = 'D:\\qanda\\infobot.pdf'

def load_docs(directory):
    loader = PyMuPDFLoader(directory)
    documents = loader.load()
    return documents

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  
    return text.strip() 

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    cleaned_documents = []    
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_documents.append(doc.__class__(cleaned_content)) 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(cleaned_documents)
    return split_documents

documents = load_docs(directory)
doc = split_docs(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
query_ret = embeddings.embed_query("Hello world")

api_key=os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key, environment="us-east-1")

index_name = "langchainqanda"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=len(query_ret),
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index_description = pc.describe_index(index_name)
host = index_description['host']

index = pc.Index(index_name=index_name,host=host)
lang_index = LangchainPinecone.from_documents(doc, embeddings, index_name=index_name)

def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = lang_index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = lang_index.similarity_search(query, k=k)
    return similar_docs

model = SentenceTransformer('all-MiniLM-L6-v2')

#query = "Summary of"
#similar_docs = get_similar_docs(query)
#print(similar_docs)