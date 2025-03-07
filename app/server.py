import os
import logging
import asyncio

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

import pg8000
from google.cloud.sql.connector import Connector
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.pgvector import PGVector

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI
app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/playground")

# (1) Initialize VectorStore
connector = Connector()

def getconn() -> pg8000.dbapi.Connection:
    """Establish a database connection."""
    return connector.connect(
        os.getenv("DB_INSTANCE_NAME", ""),
        "pg8000",
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", ""),
    )

vectorstore = PGVector(
    connection_string="postgresql+pg8000://",
    use_jsonb=True,
    engine_args=dict(creator=getconn),
    embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@003")
)

# (2) Initialize VectorStore Retriever
retriever = vectorstore.as_retriever()

# (3) Initialize BigQuery Client
client = bigquery.Client()

async def get_response(instructions):
    """Fetch responses from BigQuery based on retrieved document IDs."""
    ids = [instruction.page_content.split(' ')[0] for instruction in instructions]
    id_str = ', '.join(f"'{id}'" for id in ids)

    query = f"""
    SELECT response
    FROM astral-field-448905-v3.customer_support.customer_data
    WHERE id IN ({id_str})
    """

    loop = asyncio.get_running_loop()
    query_job = await loop.run_in_executor(None, client.query, query)  # Execute query asynchronously
    rows = await loop.run_in_executor(None, query_job.result)  # Fetch results

    responses = [row['response'] for row in rows]
    logging.debug(f"Fetched Responses: {responses}")

    return responses

# Combine retriever and response fetching
responses = retriever | RunnableLambda(get_response)

# (4) Define Prompt Template
prompt_template = PromptTemplate.from_template(
    """You are a helpful assistant and help answer customer's query and request. Answer on the basis of provided context only.

On the basis of the input customer query determine or suggest the following things about the input query: {query}:
    1. Urgency of the query based on the input query on a scale of 1-5 where 1 is least urgent and 5 is most urgent. Only tell me the number.
    2. Categorize the input query into sales, product, operations etc. Only tell me the category.
    3. Generate 1 best humble response to the input query which is similar to examples in the python list: {responses} from the internal database and is helpful to the customer.
    If the input query from customer is not clear then ask a follow-up question.
"""
)

# (5) Initialize LLM
llm = VertexAI(model_name="gemini-1.0-pro-001", temperature=0.2)

# (6) Chain Everything Together
chain = (
    RunnableParallel({
        "responses": responses,
        "query": RunnablePassthrough()
    })
    | prompt_template
    | llm
    | StrOutputParser()
)

# Add logging for LLM responses
def log_llm_response(response):
    logging.debug(f"LLM Response: {response}")
    return response

# Attach logging to the chain
chain_with_logging = chain | log_llm_response

# Expose API endpoints
add_routes(app, chain_with_logging)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




# logging.basicConfig(level=logging.DEBUG)
#
# app = FastAPI()
#
# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/playground")
#
# # (1) Initialize VectorStore
# connector = Connector()
#
# def getconn() -> pg8000.dbapi.Connection:
#     conn: pg8000.dbapi.Connection = connector.connect(
#         os.getenv("DB_INSTANCE_NAME", ""),
#         "pg8000",
#         user=os.getenv("DB_USER", ""),
#         password=os.getenv("DB_PASS", ""),
#         db=os.getenv("DB_NAME", ""),
#     )
#     return conn
#
# vectorstore = PGVector(
#     connection_string="postgresql+pg8000://",
#     use_jsonb=True,
#     engine_args=dict(
#         creator=getconn,
#     ),
#     embedding_function=VertexAIEmbeddings(
#         model_name="textembedding-gecko@003"
#     )
# )
#
# # (2) Build retriever
# client = bigquery.Client()
# def get_response(instructions):
#     ids = []
#     for instruction in instructions:
#         id = instruction.page_content.split(' ')[0]
#         ids.append(id)
#
#     id_str = ', '.join(f"'{id}'" for id in ids)
#
#     query = """
#     SELECT
#       `response`
#     FROM
#       `astral-field-448905-v3.customer_support.customer_data`
#     WHERE id IN ({id_str})
#     """
#
#     query_job = client.query(query)  # Execute query
#     rows = query_job.result()
#     responses = [row['response'] for row in rows]
#     logging.debug(f"Responses: {responses}")
#     return responses
#     # return "\n\n".join(instruction.page_content for instruction in instructions)
#
# responses = vectorstore.as_retriever() | get_response
#
# # (3) Create prompt template
# prompt_template = PromptTemplate.from_template(
#     """You are a helpful assistant and help answer customer's query and request. Answer on the basis of provided context only.
#
# On the basis of the input customer query determine or suggest the following things about the input query: {query}:
#     1. Urgency of the query based on the input query on a scale of 1-5 where 1 is least urgent and 5 is most urgent. Only tell me the number.
#     2. Categorize the input query into sales, product, operations etc. Only tell me the category.
#     3. Generate 1 best humble response to the input query which is similar to examples in the python list: {responses} from the internal database and is helpful to the customer.
#     If the input query from customer is not clear then ask a follow-up question.
#     Your answer:
# """
# )
#
# # (4) Initialize LLM
# llm = VertexAI(
#     model_name="gemini-1.0-pro-001",
#     temperature=0.2,
#     # max_output_tokens=100,
#     # top_k=40,
#     # top_p=0.95
# )
#
# # (5) Chain everything together
# chain = (
#     RunnableParallel({
#         "responses": responses,
#         "query": RunnablePassthrough()
#     })
#     | prompt_template
#     | llm
#     | StrOutputParser()
# )
#
# # Add logging for LLM responses
# def log_llm_response(response):
#     logging.debug(f"LLM Response: {response}")  # Log the LLM response
#     return response
#
# # Add the logging step to the chain
# chain_with_logging = chain | log_llm_response
#
# add_routes(app, chain_with_logging)
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)

