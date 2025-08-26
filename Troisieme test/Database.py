# import basics
import os
from dotenv import load_dotenv

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load pdf docs from folder 'documents'
with open("livredor_large_verbose.txt", encoding="utf-8") as file:
    lines = [l.strip() for l in file if l.strip()]

entries, current = [], {}
for line in lines:
    if line.startswith("Problème"):
        current = {"problem": line}
    elif line.startswith("Solution"):
        current["solution"] = line
    elif line.startswith("Mots clés"):
        current["keywords"] = line
        entries.append(current)
        current = {}


# store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)