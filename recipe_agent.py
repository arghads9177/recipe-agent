# Import libraries
import os
from dotenv import load_dotenv
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.recursive import RecursiveChunking
from agno.models.groq import Groq
from agno.agent import Agent, AgentKnowledge
from textwrap import dedent

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define Knowledgebase
pdf_knowledge_base = AgentKnowledge(
    # Table name: ai.pdf_documents
    vector_db=ChromaDb(
        collection="recipes",
        path = "./data/recipes",
        embedder=OpenAIEmbedder(),
        persistent_client=True
    )
)
# Define Agent
agent = Agent(
    model= Groq(id="llama-3.3-70b-versatile", temperature=0),
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    description= dedent("""\
        You are a helpful Recipe Assistant specialized in providing recipes for Chinese cuisine. 
        Your task is to search the knowledgebase and retrieve the most relevant recipe for the user’s query and use this as context.

        Your answer style is:
        - Clear and authoritative
        - Engaging but professional
        - Answer must be specific to the question.\
    """),
    instructions=dedent("""\
        Follow these rules strictly:

        Search the knowldegebse for recipes and provide the best match if it belongs to Chinese cuisine.
        If the recipe is not found in the knowldegebse, respond with:
        "I don't have the answer related to this recipe."
        
        If the query is not related to any recipe (e.g., general questions or off-topic queries), politely respond with:
        "I'm here to help with Chinese recipes. If you need a recipe, feel free to ask!"
        Stay polite, concise, and on-topic while assisting users.\
    """),
    markdown=True,
    debug_mode=True
)

agent.print_response("What ingredients are nedded for Chilli Paneer?")