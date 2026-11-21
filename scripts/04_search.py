
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# Load the vector database, embedding model, and local dataset
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = 512

df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


def encode_query(query):
    # Use the same model and normalization as for documents
    return model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)


def print_pinecone_results(response):
    # Display search results with paper metadata
    if not response.matches:
        print("No matching papers found.")
        return

    for position, match in enumerate(response.matches, start=1):
        metadata = match.metadata or {}

        print(f"\n{position}. {metadata.get('title', 'Unknown title')}")
        print(f"   Score: {match.score:.4f}")
        print(f"   Category: {metadata.get('category', 'unknown')}")
        print(f"   Year: {metadata.get('year', 'unknown')}")
        print(f"   arXiv ID: {metadata.get('arxiv_id', 'unknown')}")
        print(f"   Abstract: {metadata.get('abstract', '')[:200]}...")


def semantic_search(query, metadata_filter=None):
    # Search Pinecone using a normalized query embedding
    vector = encode_query(query)

    response = index.query(
        vector=vector.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        filter=metadata_filter
    )

    return response


def print_local_results(scores, metric, higher_is_better=True):
    # Sort by similarity descending or distance ascending
    if higher_is_better:
        indices = np.argsort(-scores)[:TOP_K]
    else:
        indices = np.argsort(scores)[:TOP_K]

    result_ids = []

    print(f"\n--- {metric} ---")

    for rank, i in enumerate(indices, start=1):
        paper = df.iloc[i]
        result_ids.append(str(paper["id"]))

        print(f"{rank}. {paper['title']}")
        print(f"   Score: {scores[i]:.6f}")
        print(f"   Category: {paper['category']}, Year: {paper['year']}")

    return result_ids


def compare_metrics(query):
    print("\n=== Task 3: Similarity metrics ===")
    print(f"Query: {query}")

    # Load document embeddings
    embeddings = np.load("embeddings/embeddings.npy")

    if embeddings.shape != (len(df), 768):
        raise ValueError("Unexpected embedding dimensions")

    query_vector = encode_query(query)

    # Dot product
    dot_scores = embeddings @ query_vector

    # Cosine similarity
    document_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)

    cosine_scores = dot_scores / (document_norms * query_norm)

    # Euclidean (L2) distance
    l2_scores = np.linalg.norm(
        embeddings - query_vector,
        axis=1
    )

    # Print the top 5 for each metric
    cosine_ids = print_local_results(
        cosine_scores, "Cosine similarity"
    )

    dot_ids = print_local_results(
        dot_scores, "Dot product"
    )

    l2_ids = print_local_results(
        l2_scores, "L2 distance", higher_is_better=False
    )

    # Compare rankings
    print("\n--- Metric comparison ---")
    print(f"Cosine vs Dot top-5 equal: {cosine_ids == dot_ids}")
    print(f"Cosine vs L2 top-5 equal: {cosine_ids == l2_ids}")

    print(
        "Maximum cosine-dot score difference: "
        f"{np.max(np.abs(cosine_scores - dot_scores)):.8f}"
    )

    print(
        "Query vector norm: "
        f"{np.linalg.norm(query_vector):.6f}"
    )


def main():
    print(f"Loaded local papers: {len(df)}")

    # Task 1: Pure semantic search
    query = "teaching machines to recognize objects in pictures"

    print("\n=== Task 1: Semantic search ===")
    print(f"Query: {query}")

    results = semantic_search(query)
    print_pinecone_results(results)

    # Task 2A: Recent machine learning research
    recent_query = "reinforcement learning algorithms"

    print("\n=== Task 2A: Recent cs.LG papers ===")
    print(f"Query: {recent_query}")

    recent_filter = {
        "year": {"$gte": 2022},
        "category": {"$eq": "cs.LG"}
    }

    results = semantic_search(recent_query, recent_filter)
    print_pinecone_results(results)

    # Task 2B: Older research, without category restrictions
    print("\n=== Task 2B: Papers before 2015 ===")
    print(f"Query: {recent_query}")

    old_filter = {
        "year": {"$lt": 2015}
    }

    results = semantic_search(recent_query, old_filter)
    print_pinecone_results(results)

    # Task 3: Local similarity metric comparison
    compare_metrics(query)


if __name__ == "__main__":
    main()
