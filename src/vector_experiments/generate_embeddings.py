import os
import argparse
import pickle
import pathlib
from src.vector_experiments.data_loader import load_dbpedia_data
from src.vector_experiments.models import SpladeEmbedder, BGEM3Embedder, BM25Embedder, Word2VecEmbedder, MiniLMEmbedder

def save_embeddings(embeddings, artifact, model_name, output_dir):
    """Saves embeddings and optional artifacts."""
    output_path = os.path.join(output_dir, f"{model_name}_embeddings.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved {model_name} embeddings to {output_path}")
    
    if artifact:
        artifact_path = os.path.join(output_dir, f"{model_name}_artifact.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved {model_name} artifact to {artifact_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for DBpedia dataset.")
    parser.add_argument("--model", type=str, choices=["splade", "bge-m3", "bm25", "word2vec", "minilm", "bge-m3-all", "all"], default="all", help="Model to use for embedding generation")
    parser.add_argument("--output_dir", type=str, default="data/embeddings", help="Directory to save embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents for testing")
    args = parser.parse_args()

    # Ensure output directory exists
    # If path is relative, make it absolute relative to project root
    if not os.path.isabs(args.output_dir):
        project_root = pathlib.Path(__file__).parent.parent.parent
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)

    print("Loading DBpedia dataset...")
    corpus, _, _ = load_dbpedia_data()
    
    if args.limit:
        print(f"Limiting to first {args.limit} documents.")
        corpus = {k: corpus[k] for k in list(corpus.keys())[:args.limit]}

    if args.model == "all":
        models_to_run = ["splade", "bge-m3", "bm25", "word2vec", "minilm", "bge-m3-all"]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        print(f"\n--- Processing {model_name.upper()} ---")
        embedder = None
        
        if model_name == "splade":
            embedder = SpladeEmbedder()
        elif model_name == "bge-m3":
            embedder = BGEM3Embedder(return_all=False)
        elif model_name == "bge-m3-all":
            # Using same class but with all outputs
            embedder = BGEM3Embedder(return_all=True)
        elif model_name == "bm25":
            embedder = BM25Embedder()
        elif model_name == "word2vec":
            embedder = Word2VecEmbedder()
        elif model_name == "minilm":
            embedder = MiniLMEmbedder()
            
        if embedder:
            embeddings, artifact = embedder.encode(corpus)
            save_embeddings(embeddings, artifact, model_name, output_dir)
            
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
