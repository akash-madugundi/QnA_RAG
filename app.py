import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
import os
import faiss
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

Settings.llm = Gemini(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004", api_key=GOOGLE_API_KEY, embed_batch_size=100
)
Settings.node_parser = SimpleNodeParser.from_defaults()

documents = SimpleDirectoryReader("data").load_data()
nodes = Settings.node_parser.get_nodes_from_documents(documents)
embedding_dim = 768

faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.TREE_SUMMARIZE)
query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

def calculator_tool(query: str) -> str:
    try:
        expression = query.lower().split('calculate', 1)[1].strip()
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {e}"

def dictionary_tool(query: str) -> str:
    try:
        term = query.lower().split('define', 1)[1].strip()
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}")
        if response.status_code == 200:
            data = response.json()
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return definition
        else:
            return "Definition not found."
    except Exception as e:
        return f"Error fetching definition: {e}"

st.set_page_config(page_title="RAG-Powered Multi-Agent Q&A Assistant")
st.title("RAG-Powered Multi-Agent Q&A Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "calculate" in prompt.lower():
        decision = "Calculator Tool"
        response = calculator_tool(prompt)
        context = None
    elif "define" in prompt.lower():
        decision = "Dictionary Tool"
        response = dictionary_tool(prompt)
        context = None
    else:
        decision = "RAG Pipeline"
        rag_response = query_engine.query(prompt)
        response = rag_response.response
        context = rag_response.source_nodes

    with st.chat_message("assistant"):
        st.markdown(f"**Decision:** {decision}")
        st.markdown(f"**Answer:** {response}")

    st.session_state.messages.append({"role": "assistant", "content": response})