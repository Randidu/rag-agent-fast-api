from pprint import pprint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

loader = PyPDFLoader('hr_manual.pdf',mode="page")

# embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#select your embedding
text_splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

text_split = text_splitter.split_documents(loader.load())

vector_store = InMemoryVectorStore(embedding=embedding_model)
vector_store.add_documents(documents=text_split)

llm_model = ChatOllama(model="llama3.2:1b")

def retrieve_context(user_query):
    retrieve_document = vector_store.similarity_search(user_query, k=3)
    content = "\n,\n".join(
        (f"Source:{doc.metadata}\n Content : {doc.page_content}")
        for doc in retrieve_document
    )
    return content

prompt_template = ChatPromptTemplate.from_messages([
    ("system","""You are a helpful assistant who provide answer using the provides context.
                 Use only the information from the context to answer.
                 If context doesnt have the answer say so"""),
    ("human","context :\n {context} \n\n Question : {question}")

])
rag_chain = prompt_template | llm_model

while True:
    user_question= input("\n Ask me any question...")
    resp = rag_chain.invoke({"context": retrieve_context(user_question), "question": user_question})
    pprint(resp)