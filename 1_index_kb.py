import os
import sys
import argparse
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup batch addition with retry logic to handle potential network or load issues
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def add_to_collection(collection, ids, documents, metadatas):
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

def main():
    parser = argparse.ArgumentParser(description="Index Amazon Customer Support Knowledge Base into ChromaDB.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the production_rag_knowledge_base.csv file")
    parser.add_argument("--reset", action="store_true", help="Reset/overwrite the existing ChromaDB collection")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for inserting documents into ChromaDB")
    args = parser.parse_args()

    # 1. Determine Data Path
    if args.data_path:
        data_path = args.data_path
    else:
        # Fallback logic
        if os.path.exists("./data/processed/production_rag_knowledge_base.csv"):
            data_path = "./data/processed/production_rag_knowledge_base.csv"
        elif os.path.exists("./production_rag_knowledge_base.csv"):
            data_path = "./production_rag_knowledge_base.csv"
        elif os.path.exists("./twcs/production_rag_knowledge_base.csv"):
            data_path = "./twcs/production_rag_knowledge_base.csv"
        else:
            print("Error: Could not find production_rag_knowledge_base.csv. Please specify --data_path.")
            sys.exit(1)
            
    print(f"Loading data from: {data_path}")

    # 2. Data Ingestion & Validation
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    required_columns = {"tweet_id", "rag_context_text"}
    if not required_columns.issubset(df.columns):
        print(f"Error: CSV must contain columns {required_columns}. Found: {list(df.columns)}")
        sys.exit(1)

    initial_count = len(df)
    # Drop rows with null or empty text records
    df = df.dropna(subset=["tweet_id", "rag_context_text"])
    df = df[df["rag_context_text"].str.strip() != ""]
    print(f"Loaded {len(df)} valid records (dropped {initial_count - len(df)} null/empty records).")

    # 3. Vector Store Architecture Setup
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "amazon_support_kb"

    if args.reset:
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Reset: Deleted existing collection '{collection_name}'.")
        except Exception:
            # Collection might not exist
            pass

    # 4. Configure Embedding Function
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    print(f"Using embedding model: {model_name}")
    
    # Use the sentence transformer embedding function
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"} # Set cosine similarity
    )

    # 5. Ingestion in Batches
    records = df.to_dict("records")
    
    print(f"Starting ingestion in batches of {args.batch_size}...")
    for i in tqdm(range(0, len(records), args.batch_size)):
        batch = records[i:i + args.batch_size]
        
        ids = [str(row["tweet_id"]) for row in batch]
        documents = [str(row["rag_context_text"]) for row in batch]
        metadatas = [{"tweet_id": str(row["tweet_id"])} for row in batch]
        
        try:
            add_to_collection(collection, ids=ids, documents=documents, metadatas=metadatas)
        except Exception as e:
            print(f"\nFailed to add batch starting at index {i}. Error: {e}")
            sys.exit(1)

    print("Ingestion complete!")

    # 6. Verification
    total_docs = collection.count()
    print(f"\n--- Verification ---")
    print(f"Total documents in collection '{collection_name}': {total_docs}")

    test_query = "Where is my package?"
    print(f"\nRunning test similarity query: '{test_query}'")
    
    results = collection.query(
        query_texts=[test_query],
        n_results=3
    )

    if results and "documents" in results and results["documents"]:
        print("\nTop-3 retrieved contexts:")
        for idx, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][idx]
            distance = results["distances"][0][idx] if "distances" in results and results["distances"] else "N/A"
            print(f"\nResult {idx + 1} [Tweet ID: {metadata.get('tweet_id')}] (Distance: {distance:.4f}):")
            print(f"{doc}")
    else:
        print("Verification failed: No results returned from test query.")

if __name__ == "__main__":
    main()
