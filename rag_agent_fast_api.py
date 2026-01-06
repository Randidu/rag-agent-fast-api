from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI
from pydantic import BaseModel
from starlette import status
from starlette.exceptions import HTTPException

load_dotenv()
app = FastAPI(title="HR RAG Agent")

class QueryRequest(BaseModel):
    question :str

class QueryResponse(BaseModel):
    question :str
    answer : str
    status : str = "Success"

# loader = PyPDFLoader('hr_manual.pdf',mode="page")
# embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#select your embedding
text_splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

# text_split = text_splitter.split_documents(loader.load())

vector_store = PineconeVectorStore(embedding=embedding_model,index_name="hrdocs3",namespace="hr-document")
# vector_store.add_documents(documents=text_split)

llm_model = ChatOpenAI(model="gpt-5.2-2025-12-11")
@tool
def enhance_user_query(user_query):
    """
    Enhance user query using an LLM for better relative
    Args :
        user_query: The user's optimized for semantic search
    """
    enhance_prompt = ChatPromptTemplate.from_template(
        """
        You are query optimization expert for an HR Knowledge base.
        you task is to expand users query to improved document retrieval
        
        original query : {original_query}
        
        
        Instructions:
        -Expand abbreviations (PTO- paid time off, HR- human resource ,etc)
        -Include relevant keyword for HR context
        -Maintain original intent
        
        Retune only enhance query nothing else   
        """
    )
@tool
def write_result_to_file(content:str,file_name:str="query_result.txt"):
    """
    Write query result to a text file for persistent storage

    Agrs:
        content : The result content to write
        filename : target file name (default : query-result.txt)
    retunes:
        confirmation message of successfully write
    """
    try:
        with open(file_name,"a") as f:
            f.write(content)

        return f"Successfully wrote contents to file name -{file_name}"
    except Exception as e:
        return f"error write to file {str(e)}"

@tool(response_format="content_and_artifact")
def retrieve_context(enhanced_user_query):
    """Retrieve information from vector database to help answer enhance user query queries
    retune from enhance_user_query"""
    retrieve_document = vector_store.similarity_search(enhanced_user_query, k=3)
    content = "\n,\n".join(
        (f"Source:{doc.metadata}\n Content : {doc.page_content}")
        for doc in retrieve_document
    )
    return content,retrieve_document

tools = [retrieve_context,enhance_user_query,write_result_to_file]

prompt = """
                You have access to tools that enhances user query and retrieves context from hr documents.
                You also have access to a tool which allows to write output content to a file
                use it when required tool name - write_results_to_file args: (content: last_message from AI, filename: name of the file)
                Use the tool to help answer user queries
"""


agent = create_agent(model=llm_model, tools=tools, system_prompt=prompt)

@app.post("/query",response_model=QueryResponse)
def query(request:QueryRequest):
    try:
        resp = agent.invoke({"messages": [{"role": "user", "content": request.question}]})
        print(resp)
        answer = answer = getattr(resp["messages"][-1], "content", None)
        return QueryResponse(
            question=request.question,
            answer=answer,
            status="Success"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=f"error occurred : {str(e)}")


