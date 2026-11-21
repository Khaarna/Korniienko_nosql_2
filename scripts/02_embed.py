
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/arxiv_subset.parquet")
OUTPUT_FILE = Path("embeddings/embeddings.npy")

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64
EXPECTED_DIM = 768


def main():
    # Load prepared papers
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded papers: {len(df)}")

    # Combine title and abstract using the required separator
    texts = (
        df["title"] + " [SEP] " + df["abstract"]
    ).tolist()

    # Load the embedding model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Limit input length to the model's supported sequence length
    model.max_seq_length = 512

    # Generate normalized embeddings in batches
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(np.float32)

    # Validate the embeddings
    print("\nEmbedding results:")
    print(f"Processed texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    first_norm = np.linalg.norm(embeddings[0])
    print(f"First embedding norm: {first_norm:.6f}")

    if embeddings.shape != (len(df), EXPECTED_DIM):
        raise ValueError("Unexpected embedding dimensions")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain invalid values")

    if not np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        atol=1e-4
    ):
        raise ValueError("Embeddings are not normalized")

    # Save embeddings for Pinecone upload
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)

    print(f"\nSaved embeddings to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
