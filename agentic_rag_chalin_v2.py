from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

# loader = PyPDFLoader('hr_manual.pdf',mode="page")

# embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#select your embedding
text_splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

# text_split = text_splitter.split_documents(loader.load())

vector_store = PineconeVectorStore(embedding=embedding_model,index_name="hrdocs3",namespace="hr-document")
# vector_store.add_documents(documents=text_split)

llm_model = ChatOpenAI(model="gpt-4-turbo")

@tool
def retrieve_context(user_query):
    """Retrieve information from vector database to help answer user queries"""
    retrieve_document = vector_store.similarity_search(user_query, k=3)
    content = "\n,\n".join(
        (f"Source:{doc.metadata}\n Content : {doc.page_content}")
        for doc in retrieve_document
    )
    return content

tools = [retrieve_context]

prompt = """
Your have access to a tool that retrieves context from hr documents.
Use the tool to help answer uer queries
"""

agent = create_agent(model=llm_model, tools=tools, system_prompt=prompt)

while True:
    question =input("\nAsk me anything :")
    resp = agent.invoke({"messages" : [{"role" : "user", "content" : question}]})
    print(resp["messages"][-1].content)


