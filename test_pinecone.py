import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
print(f"API Key: {api_key[:5]}...")

pc = Pinecone(api_key=api_key)

# List all indexes
print("\nListing all indexes...")
print(f"Available indexes: {pc.list_indexes()}")

# Connect to index
index_name = "rag-test"
index = pc.Index(index_name)
print(f"\nConnected to index: {index_name}")

# Describe index
print("\nDescribing index...")
print(f"Index description: {index.describe_index_stats()}")

# Test vector
test_vector = [0.1] * 1024  # 1024-dimensional vector
test_metadata = {"text": "This is a test vector"}

# Upsert test vector
print("\nUpserting test vector...")
upsert_response = index.upsert(
    vectors=[{
        "id": "test_vec",
        "values": test_vector,
        "metadata": test_metadata
    }],
    namespace="test"
)
print(f"Upsert response: {upsert_response}")

# Check stats
print("\nChecking index stats...")
stats = index.describe_index_stats()
print(f"Stats: {stats}")

# Query test
print("\nQuerying test vector...")
query_response = index.query(
    vector=test_vector,
    top_k=1,
    namespace="test",
    include_metadata=True
)
print(f"Query results: {query_response}")

# Cleanup
print("\nCleaning up...")
index.delete(deleteAll=True, namespace="test")
print("Cleanup successful")
