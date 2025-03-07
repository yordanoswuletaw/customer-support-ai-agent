import os
from google.cloud.sql.connector import Connector
import pg8000
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import bigquery


# Retrieve all Cloud Run release notes from BigQuery
client = bigquery.Client()
query = """
SELECT
  `instruction`
FROM
  `astral-field-448905-v3.customer_support.customer_data` 
"""
rows = client.query(query)
instruction = [row['instruction'] for row in rows]

# Set up a PGVector instance
connector = Connector()

def getconn() -> pg8000.dbapi.Connection:
    conn: pg8000.dbapi.Connection = connector.connect(
        os.getenv("DB_INSTANCE_NAME", ""),
        "pg8000",
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", ""),
    )
    return conn

store = PGVector(
    connection_string="postgresql+pg8000://",
    use_jsonb=True,
    engine_args=dict(
        creator=getconn,
    ),
    embedding_function=VertexAIEmbeddings(
        model_name="textembedding-gecko@003"
    ),
    pre_delete_collection=True
)

# Save all instruction into the Cloud SQL database
batch_size = 1000  # Adjust as needed
for i in range(0, len(instruction), batch_size):
    batch = instruction[i : i + batch_size]
    store.add_texts(batch)

print(f"Done saving!")