
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


load_dotenv()

INPUT_PARQUET = Path("data/arxiv_subset.parquet")
INPUT_EMBEDDINGS = Path("embeddings/embeddings.npy")

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200


def main():
    # Load data and embeddings
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    print(f"Loaded papers: {len(df)}")
    print(f"Embedding shape: {embeddings.shape}")

    if embeddings.shape != (len(df), VECTOR_DIM):
        raise ValueError("Dataset and embeddings do not match")

    # Connect to Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Create the index if it does not exist
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"\nCreating index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print(f"\nIndex already exists: {INDEX_NAME}")

    # Verify the index configuration
    description = pc.describe_index(INDEX_NAME)

    if description.dimension != VECTOR_DIM:
        raise ValueError("Index dimension does not match")

    if description.metric != "cosine":
        raise ValueError("Index metric must be cosine")

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("Waiting for index to become ready...")
        time.sleep(5)

    index = pc.Index(INDEX_NAME)

    # Upload vectors with paper metadata
    print("\nUploading vectors...")

    for start in tqdm(
        range(0, len(df), BATCH_SIZE),
        desc="Uploading batches"
    ):
        end = min(start + BATCH_SIZE, len(df))
        vectors = []

        for i in range(start, end):
            paper = df.iloc[i]

            metadata = {
                "arxiv_id": str(paper["id"]),
                "title": str(paper["title"]),
                "abstract": str(paper["abstract"])[:500],
                "authors": str(paper["authors"])[:200],
                "year": int(paper["year"]),
                "category": str(paper["category"])
            }

            vectors.append({
                "id": f"paper_{i}",
                "values": embeddings[i].tolist(),
                "metadata": metadata
            })

        index.upsert(vectors=vectors)

    # Display index statistics
    stats = index.describe_index_stats()

    print("\nUpload completed.")
    print(f"Index name: {INDEX_NAME}")
    print(f"Total vectors reported: {stats.total_vector_count}")
    print(f"Expected vectors: {len(df)}")
    print("Note: Pinecone statistics may update asynchronously.")


if __name__ == "__main__":
    main()
