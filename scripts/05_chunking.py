
import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

MAX_WORDS = 120
OVERLAP_WORDS = 30
BATCH_SIZE = 32
UPLOAD_BATCH_SIZE = 100
TOP_K = 5

INDEX_NAMES = {
    "fixed": "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic"
}

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = 512


def fixed_size_chunking(text):
    # Split text into overlapping word windows
    words = text.split()
    chunks = []
    step = MAX_WORDS - OVERLAP_WORDS

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + MAX_WORDS])

        if chunk:
            chunks.append(chunk)

        if start + MAX_WORDS >= len(words):
            break

    return chunks


def semantic_chunking(text):
    # Split text at sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = []
    current_size = 0

    for sentence in sentences:
        words = sentence.split()

        if not words:
            continue

        # Handle sentences longer than the maximum chunk size
        if len(words) > MAX_WORDS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_size = 0

            for start in range(0, len(words), MAX_WORDS):
                chunks.append(
                    " ".join(words[start:start + MAX_WORDS])
                )
            continue

        if current_size + len(words) > MAX_WORDS:
            chunks.append(" ".join(current))
            current = []
            current_size = 0

        current.extend(words)
        current_size += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def create_index(name):
    # Create a serverless index if it does not exist
    if name not in pc.list_indexes().names():
        print(f"Creating index: {name}")

        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    description = pc.describe_index(name)

    if description.dimension != VECTOR_DIM:
        raise ValueError(f"Wrong dimension for {name}")

    if description.metric != "cosine":
        raise ValueError(f"Wrong metric for {name}")

    while not pc.describe_index(name).status["ready"]:
        time.sleep(5)

    return pc.Index(name)


def prepare_chunks(df, strategy):
    # Generate chunks and preserve paper metadata
    records = []

    chunk_function = (
        fixed_size_chunking
        if strategy == "fixed"
        else semantic_chunking
    )

    for paper_idx, paper in df.iterrows():
        chunks = chunk_function(paper["abstract"])

        for chunk_idx, chunk in enumerate(chunks):
            records.append({
                "id": f"{strategy}_{paper_idx}_{chunk_idx}",
                "arxiv_id": str(paper["id"]),
                "title": str(paper["title"]),
                "text": chunk,
                "chunk_number": chunk_idx,
                "year": int(paper["year"]),
                "category": str(paper["category"])
            })

    return records


def upload_chunks(index, records, strategy):
    # Add the title and separator to each chunk
    texts = [
        record["title"] + " [SEP] " + record["text"]
        for record in records
    ]

    print(f"\nGenerating {strategy} embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    if embeddings.shape != (len(records), VECTOR_DIM):
        raise ValueError("Unexpected embedding dimensions")

    print(f"Uploading {len(records)} chunks...")

    for start in tqdm(
        range(0, len(records), UPLOAD_BATCH_SIZE),
        desc=f"Uploading {strategy}"
    ):
        end = min(start + UPLOAD_BATCH_SIZE, len(records))
        vectors = []

        for i in range(start, end):
            record = records[i]

            vectors.append({
                "id": record["id"],
                "values": embeddings[i].tolist(),
                "metadata": {
                    "arxiv_id": record["arxiv_id"],
                    "title": record["title"],
                    "chunk_text": record["text"][:1500],
                    "chunk_number": record["chunk_number"],
                    "year": record["year"],
                    "category": record["category"]
                }
            })

        index.upsert(vectors=vectors)

    print(f"Upload completed: {strategy}")


def search_chunks(index, query):
    # Search for the most relevant chunks
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    response = index.query(
        vector=vector.tolist(),
        top_k=TOP_K,
        include_metadata=True
    )

    results = []

    for rank, match in enumerate(response.matches, start=1):
        metadata = match.metadata or {}

        results.append(metadata.get("arxiv_id"))

        print(f"\n{rank}. {metadata.get('title')}")
        print(f"   Score: {match.score:.4f}")
        print(f"   arXiv ID: {metadata.get('arxiv_id')}")
        print(f"   Chunk: {metadata.get('chunk_number')}")
        print(f"   Text: {metadata.get('chunk_text', '')[:180]}...")

    return results


def print_chunk_statistics(records, strategy):
    lengths = [len(record["text"].split()) for record in records]

    print(f"\n=== {strategy.upper()} CHUNKING ===")
    print(f"Total chunks: {len(records)}")
    print(f"Average words per chunk: {np.mean(lengths):.1f}")
    print(f"Minimum chunk size: {min(lengths)}")
    print(f"Maximum chunk size: {max(lengths)}")

    print("\nExample chunk:")
    print(records[0]["text"][:350])


def main():
    df = pd.read_parquet(
        "data/arxiv_subset.parquet"
    ).copy()

    # Select 30 papers with the longest abstracts
    df["word_count"] = df["abstract"].str.split().str.len()

    longest = (
        df.nlargest(30, "word_count")
        .reset_index(drop=True)
    )

    print(f"Selected papers: {len(longest)}")
    print(f"Longest abstract: {longest['word_count'].max()} words")
    print(f"Shortest selected: {longest['word_count'].min()} words")

    indexes = {}

    for strategy, index_name in INDEX_NAMES.items():
        records = prepare_chunks(longest, strategy)

        print_chunk_statistics(records, strategy)

        index = create_index(index_name)
        upload_chunks(index, records, strategy)

        indexes[strategy] = index

    # Use identical queries for both strategies
    test_queries = [
        "methods and algorithms for scientific research",
        "statistical models and analysis of experimental data",
        longest.iloc[0]["title"]
    ]

    for query in test_queries:
        print(f"\n\n=== QUERY: {query} ===")

        for strategy, index in indexes.items():
            print(f"\n--- {strategy.upper()} SEARCH ---")
            search_chunks(index, query)

    # Display final index statistics
    print("\n=== FINAL INDEX STATISTICS ===")

    for strategy, index in indexes.items():
        stats = index.describe_index_stats()

        print(
            f"{strategy}: {stats.total_vector_count} vectors "
            "(statistics may update asynchronously)"
        )


if __name__ == "__main__":
    main()
