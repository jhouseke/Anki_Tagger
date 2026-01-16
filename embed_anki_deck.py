import os
import argparse

import pandas as pd
import tiktoken
from embedding_client import EmbeddingClient, EmbeddingProvider, get_embedding
from tqdm import tqdm

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"  # Default model (OpenRouter)
EMBEDDING_ENCODING = "cl100k_base"
MAX_TOKENS = 8000  # For OpenRouter (OpenAI models)
MAX_TOKENS_OLLAMA = 512  # For Ollama models (conservative default, many have 512 token limit)

def load_dataset(input_datapath):
    assert os.path.exists(input_datapath), f"{input_datapath} does not exist. Please check your file path."

    df = pd.read_csv(input_datapath, sep='\t', header=None, usecols=[0,1], names=["guid", "card"], comment='#').dropna()
    return df

def filter_by_tokens(df, encoding, max_tokens):
    df["tokens"] = df.card.apply(lambda x: len(encoding.encode(x)))
    return df[df.tokens <= max_tokens]

def calculate_embeddings(df, embedding_client):
    return [embedding_client.get_embedding(card) for card in tqdm(df.card, desc="Calculating embeddings", dynamic_ncols=True)]

def save_embeddings(df, output_prefix):
    df.to_csv(f"./{output_prefix}_embeddings.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for Anki deck')
    parser.add_argument('--provider', type=str, default='openrouter',
                        choices=['openrouter', 'ollama'],
                        help='Embedding provider: openrouter or ollama (default: openrouter)')
    parser.add_argument('--model', type=str, default=EMBEDDING_MODEL,
                        help=f'Model name (default: {EMBEDDING_MODEL})')
    parser.add_argument('--input', type=str, default='./anki.txt',
                        help='Input file path (default: ./anki.txt)')
    parser.add_argument('--output', type=str, default='anki',
                        help='Output file prefix (default: anki)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for OpenRouter (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--ollama-url', type=str, default=None,
                        help='Ollama base URL (default: http://localhost:11434)')
    
    args = parser.parse_args()
    
    # Set deck to embed.
    # This is the deck you'll apply your tags to in the end.
    # In anki, export deck notes as plain text with GUID flag checked
    input_datapath = args.input
    output_prefix = args.output

    # Determine max_tokens based on provider
    if args.provider == 'ollama':
        max_tokens_for_filter = MAX_TOKENS_OLLAMA
        max_tokens_for_client = MAX_TOKENS_OLLAMA
    else:
        max_tokens_for_filter = MAX_TOKENS
        max_tokens_for_client = None  # Let OpenRouter handle its own limits

    # Initialize embedding client
    try:
        embedding_client = EmbeddingClient(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.ollama_url,
            max_tokens=max_tokens_for_client
        )
    except (ImportError, ValueError) as e:
        print(f"Error initializing embedding client: {e}")
        return

    # Load and preprocess dataset
    df = load_dataset(input_datapath)
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    df = filter_by_tokens(df, encoding, max_tokens_for_filter)

    # Calculate embeddings for cards
    df["emb"] = calculate_embeddings(df, embedding_client)

    # Save embeddings to file
    save_embeddings(df, output_prefix)

if __name__ == "__main__":
    main()
