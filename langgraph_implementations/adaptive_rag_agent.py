from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


class RAGState(TypedDict):
    question: str
    context: str
    is_relevant: bool
    answer: str
    enriched_question: str
    web_results: str


loader = PyPDFLoader('hr_manual.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

text_splits = text_splitter.split_documents(loader.load())

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = InMemoryVectorStore(embedding=embedding_model)

vector_store.add_documents(documents=text_splits)

model = ChatOpenAI(model="gpt-5.2-2025-12-11")


def retrieve_context(query: str):
    """Retrieve information from vector database to help answer user queries"""

    retrieve_documents = vector_store.similarity_search(query, k=5)

    content = "\n\n".join(
        f"Source : {doc.metadata} \n Content : {doc.page_content}" for doc in retrieve_documents
    )

    return content


def validate_context(question: str, context: str):
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a strict validator.
        Return ONLY 'Yes' or 'No'.
        Answer YES if the context contains information that can directly answer the question
        """),
        ("human", """
        Question : {question}

        Context : {context}""")
    ])

    validation_chain = validation_prompt | model

    result = validation_chain.invoke({
        "question": question,
        "context": context
    })

    return result.content.strip() == "YES"  # Returns a True or False boolean


def retrieve_node(state: RAGState):
    context = retrieve_context(state["enriched_question"])
    return {"context": context}


def validate_node(state: RAGState):
    is_relevant = validate_context(state["question"],
                                   state["context"])

    return {"is_relevant": is_relevant}


def answer_node(state: RAGState):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers question based on the provided context.
                    Use Only the information from the context to answer"""),
        ("human", """Context: {context}
                    Question : {question}""")
    ])

    rag_chain = prompt_template | model

    response = rag_chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": response.content}


def enrich_query_node(state: RAGState):
    enrich_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You rewrite user queries to improve document retrival.
        Make the question more specific and focused.
        Do not answer the question
        """),
        ("human", """
        Original Question : {question}
        Rewrite the question for better search
        """)
    ])

    enrich_chain = enrich_prompt | model
    enriched_results = enrich_chain.invoke({
        "question": state["question"]
    })
    return {"enriched_question": enriched_results.content}


def web_search_node(state: RAGState):
    web_search = web_search_tool.invoke({
        "query": state["question"]
    })

    results = web_search["results"]

    web_results_joined = "\n".join(
        [result["content"] for result in results]
    )

    return {"web_results": web_results_joined}


def web_answer_node(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You answer question using web search results.
        If the answer is uncertain,say so.
        """),
        ("human", """
        question : {question}

        Web Search Results:
        {web_result}""")
    ])

    chain = prompt | model

    response = chain.invoke({
        "question": state["question"],
        "web_result": state["web_results"]
    })

    return {"answer": response.content}


def is_relevant_condition(state: RAGState):
    return state["is_relevant"]


graph = StateGraph(RAGState)
graph.add_node("enrich_query_node", enrich_query_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("validate_node", validate_node)
graph.add_node("answer_node", answer_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("web_answer_node", web_answer_node)

graph.add_edge(START, "enrich_query_node")
graph.add_edge("enrich_query_node", "retrieve_node")
graph.add_edge("retrieve_node", "validate_node")

graph.add_conditional_edges(
    "validate_node",
    is_relevant_condition,
    {True: "answer_node", False: "web_search_node"}
)

graph.add_edge("web_search_node", "web_answer_node")
graph.add_edge("answer_node", END)

adaptive_rag_graph = graph.compile()

while True:
    question = input("\n Ask me any question... :")
    result = adaptive_rag_graph.invoke({
        "question": question
    })
    print(result["answer"])