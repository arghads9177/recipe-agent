# Import Libraries
import os
from dotenv import load_dotenv
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.recursive import RecursiveChunking

# Load Environment Variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

pdf_knowledge_base = PDFKnowledgeBase(
    path="./data",
    # Table name: ai.pdf_documents
    vector_db=ChromaDb(
        collection="recipes",
        path = "./data/recipes",
        embedder=OpenAIEmbedder(),
        persistent_client=True
    ),
    reader=PDFReader(chunk=True),
    chunking_strategy=RecursiveChunking(chunk_size=1000, overlap= 50)
)

pdf_knowledge_base.load(recreate=True)

print("Recipe knowledge base stored successfully!")