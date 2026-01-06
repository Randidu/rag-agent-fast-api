from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

app = FastAPI(title="HR Rang agent")


class QueryRequest(BaseModel):
    user_id: str
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    user_id: str
    message: str
    status: str


load_dotenv()
loader = PyPDFLoader('hr_manual.pdf')


embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vector_store = PineconeVectorStore(index_name="hrdocs5", embedding=embedding_model)


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
# text_splits = text_splitter.split_documents(loader.load())
#
# vector_store.add_documents(documents=text_splits)

openai_model = ChatOpenAI(model="gpt-5.2-2025-12-11")


@tool
def retrieve_context(enhanced_user_query: str):
    """Retrieve information to hel[ answer user queries using enhanced_query
    returned from the enhance_user_query tool"""
    retrieved_documents = vector_store.similarity_search(enhanced_user_query, k=5)

    content = "\n\n".join(
        (f"Source: {doc.metadata} \nContent : {doc.page_content}" for doc in retrieved_documents)
    )
    return content


@tool
def enhance_user_query(user_query: str):
    """
    Enhance user query using an LLM for better retrieval

    Args:
        user_query: The user's query

    Returns:
        Enhanced query optimized for semantic search
    """

    enhancement_prompt = ChatPromptTemplate.from_template(
        """
        You are a query optimization expert for an HR knowledge base.
        Your task is to expand users query to improve document retrieval

        Original query: {original_query}

        Instructions:
        - Expand abbreviation ( PTO - paid time off, HR - human resource, etc)
        - Include relevant keywords for HR context
        - Maintain original intent
        """
    )

    enhancement_chain = enhancement_prompt | openai_model
    resp = enhancement_chain.invoke({"original_query": user_query})
    return resp.content


@tool
def write_results_to_file(content: str, file_name: str = "leave_policy.txt"):
    """
    Write query results to a text file for persistent storage.

    Args:
        content: The result content to write
        file_name: target filename ( default : leave_policy.txt)

    returns:
     Confirmation message of successful write
    """
    try:
        with open(file_name, "a") as f:
            f.write(str(content))
            return f"successfully wrote contents to filename - {file_name}"
    except Exception as e:
        return f"Error writing to file : {str(e)}"


system_prompt = """
                You have access to tools that enhances user query and retrieves context from hr documents.
                You also have access to a tool which allows to write output content to a file
                use it when required tool name - write_results_to_file args: (content: last_message from AI, filename: name of the file)
                Use the tool to help answer user queries
                """

tools = [retrieve_context, enhance_user_query, write_results_to_file]
agent = create_agent(model=openai_model,
                     system_prompt=system_prompt,
                     tools=tools,
                     middleware=[
                         HumanInTheLoopMiddleware(
                             interrupt_on={
                                 "retrieve_context": False,
                                 "enhance_user_query": False,
                                 "write_results_to_file": True
                             },
                             description_prefix="Tool execution pending approval",
                         )],
                     checkpointer=InMemorySaver(),
                     )


class ApproveRequest(BaseModel):
    user_id: str
    decision: str

@app.post("/approve")
def human_approval_requests(request: ApproveRequest):
    config = {"configurable": {"thread_id": request.user_id}}
    result = agent.invoke(
        Command(resume={"decisions": [{"type": request.decision}]}),config=config
    )

    answer = result["messages"][-1].content if result.get("messages") else "No answer"
    return answer


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    config = {"configurable": {"thread_id": request.user_id}}
    resp = agent.invoke({"messages": [{"role": "user", "content": request.question}]},
                        config=config)

    if '__interrupt__' in resp:
        return QueryResponse(
            question=request.question,
            answer="No Answer",
            user_id=request.user_id,
            message="Write to file tool is pending human approval",
            status="Interrupted"
        )

    return QueryResponse(
        question=request.question,
        answer=resp["messages"][-1].content,
        user_id=request.user_id,
        message="Success",
        status="Success"
    )