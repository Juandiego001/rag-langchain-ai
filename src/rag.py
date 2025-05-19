import os
import tempfile
from langchain import hub
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langgraph.graph import StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
CHAT_MODEL = os.getenv('CHAT_MODEL') # Especificar modelo por variable de entorno

embeddings = OllamaEmbeddings(model='nomic-embed-text')

llm = ChatOllama(
    model = CHAT_MODEL,
    temperature = 0.0,
    num_predict = 256,
    top_p=0.5,
)

prompt = hub.pull('rlm/rag-prompt')


def retrieve(state, vector_store):
    retrieved_docs = vector_store.similarity_search(state['question'])
    return {'context': retrieved_docs}


def generate(state):
    docs_content = '\n\n'.join(doc.page_content for doc in state['context'])

    messages = [
        {'role': 'system', 'content': 'Responde siempre en español'},
        {'role': 'user', 'content': f'Pregunta: {state['question']}\n\nContexto:\n{docs_content}'},
    ]

    response = llm.invoke(messages)
    return {'answer': response.content}


class Graph(TypedDict):
    question: str
    context: List[Document]
    answer: str


def get_response_model(user_input, uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    vector_store = Chroma(
        collection_name='example_collection',
        embedding_function=embeddings,
        persist_directory='./chromadb',
    )
    _ = vector_store.add_documents(documents=split_docs)

    graph_builder = StateGraph(Graph)
    graph_builder.add_node('retrieve', lambda s: retrieve(s, vector_store))
    graph_builder.add_node('generate', generate)
    graph_builder.set_entry_point('retrieve')
    graph_builder.add_edge('retrieve', 'generate')
    graph = graph_builder.compile()

    response = graph.invoke({'question': user_input})
    return response