import os
import argparse
from pathlib import Path

import pandas as pd
import tiktoken
from embedding_client import EmbeddingClient, EmbeddingProvider, get_embedding
from tqdm import tqdm
from config_loader import (
    load_config,
    get_api_key,
    PROJECT_ROOT,
    DECK_EXPORT,
    EMBEDDINGS_CSV,
    require_file,
)

def load_dataset(input_datapath):
    df = pd.read_csv(input_datapath, sep='\t', header=None, usecols=[0,1], names=["guid", "card"], comment='#').dropna()
    return df

def filter_by_tokens(df, encoding, max_tokens):
    df["tokens"] = df.card.apply(lambda x: len(encoding.encode(x)))
    return df[df.tokens <= max_tokens]

def calculate_embeddings(df, embedding_client):
    return [embedding_client.get_embedding(card) for card in tqdm(df.card, desc="Calculating embeddings", dynamic_ncols=True)]

def save_embeddings(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    cfg = load_config()
    emb = cfg["embedding"]

    parser = argparse.ArgumentParser(description='Generate embeddings for Anki deck')
    parser.add_argument('--provider', type=str, default=emb.get("provider", "openrouter"),
                        choices=['openrouter', 'ollama'],
                        help='Embedding provider (default from config)')
    parser.add_argument('--model', type=str, default=emb.get("model"),
                        help='Embedding model (default from config)')
    parser.add_argument('--input', type=str, default=None,
                        help=f'Deck export path (default: project root / {DECK_EXPORT})')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenRouter API key (or OPENROUTER_API_KEY env)')
    parser.add_argument('--ollama-url', type=str, default=emb.get("ollama_url"),
                        help='Ollama base URL (default from config)')
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input deck export not found: {input_path}")
        input_datapath = str(input_path)
    else:
        input_datapath = str(require_file(DECK_EXPORT, "embed_anki_deck (input)"))
    output_path = PROJECT_ROOT / EMBEDDINGS_CSV
    if output_path.exists():
        print(f"Embeddings already exist: {output_path} (skipping)")
        return

    provider = args.provider
    model = args.model
    api_key = args.api_key or get_api_key(cfg)

    if provider == 'ollama':
        max_tokens_for_filter = emb.get("max_tokens_ollama", 512)
        max_tokens_for_client = max_tokens_for_filter
    else:
        max_tokens_for_filter = emb.get("max_tokens", 8000)
        max_tokens_for_client = None
    encoding_name = emb.get("encoding", "cl100k_base")

    try:
        embedding_client = EmbeddingClient(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=args.ollama_url,
            max_tokens=max_tokens_for_client
        )
    except (ImportError, ValueError) as e:
        print(f"Error initializing embedding client: {e}")
        return

    df = load_dataset(input_datapath)
    encoding = tiktoken.get_encoding(encoding_name)
    df = filter_by_tokens(df, encoding, max_tokens_for_filter)

    # Calculate embeddings for cards
    df["emb"] = calculate_embeddings(df, embedding_client)

    # Save embeddings to file in project root
    save_embeddings(df, output_path)

if __name__ == "__main__":
    main()
