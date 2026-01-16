import os, re, sys, csv, glob, time, argparse
import tiktoken
import pdfplumber
from embedding_client import EmbeddingClient, EmbeddingProvider
from pathlib import Path

try:
    from openrouter import OpenRouter
except ImportError:
    OpenRouter = None

MAX_TOKENS = 32000  # Increased for Gemini models
TOKEN_BUFFER = 1000

def get_openrouter_api_key(api_key=None):
    """Get OpenRouter API key from parameter or environment variable."""
    if OpenRouter is None:
        print("OpenRouter package is required. Install it with: pip install openrouter")
        sys.exit(1)
    api_key = api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Set your OpenRouter API key as an environment variable named 'OPENROUTER_API_KEY' eg In terminal: export OPENROUTER_API_KEY=your-api-key")
        sys.exit(1)
    return api_key

def handle_api_error(func):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle various API errors (rate limits, connection errors, etc.)
                error_str = str(e).lower()
                if 'rate limit' in error_str or '429' in error_str or 'quota' in error_str:
                    print('Rate limit exceeded. Waiting 10s before retrying.')
                    time.sleep(10)
                elif 'connection' in error_str or 'timeout' in error_str:
                    print('Connection error. Waiting 10s before retrying.')
                    time.sleep(10)
                else:
                    print(f'API Error: {e}. Waiting 10s before retrying.')
                    time.sleep(10)  # wait for 10 seconds before retrying
    return wrapper

def count_tokens(text):
    # Use cl100k_base encoding as a general estimate (works reasonably well for most models)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = list(enc.encode(text))
    return len(tokens)

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text

def generate_questions(prompt, model="google/gemini-3-flash-preview", temperature=1.0, api_key=None):
    formatted_prompt = [{"role": "system", "content": "You are a socratic medical school tutor that provides comprehensive learning questions."},
                        {"role": "user", "content": prompt}]
    remaining_tokens = MAX_TOKENS - count_tokens(" ".join([message["content"] for message in formatted_prompt])) - TOKEN_BUFFER

    if remaining_tokens < TOKEN_BUFFER:
        print(f"Warning! Input text is longer than the model can support. Consider trimming input and trying again.")
        print(f"Current length: {count_tokens(prompt)}, recommended < {MAX_TOKENS - TOKEN_BUFFER}")
        raise ValueError('Input text too long')

    # OpenRouter client must be used as a context manager
    api_key = get_openrouter_api_key(api_key=api_key)
    with OpenRouter(api_key=api_key) as openrouter:
        response = openrouter.chat.send(
            model=model,
            messages=formatted_prompt,
            max_tokens=remaining_tokens,
            temperature=temperature
        )
        # Access response inside context manager
        result = response.choices[0].message.content.strip()
    
    return result

@handle_api_error
def define_objectives_from_pdf(pdf_file, model="google/gemini-3-flash-preview", temperature=1.0, api_key=None):
    text = extract_text_from_pdf(pdf_file)
    prompt = f"Generate a list of learning questions that comprehensively covers the most important information presented in the text below to understand the topics presented.\n\n{text}"
    generated_text = generate_questions(prompt, model=model, temperature=temperature, api_key=api_key)
    objectives = [line.strip() for line in generated_text.split("\n") if line.strip()]
    return objectives

@handle_api_error
def generate_embedding(obj, embedding_client, embedding_encoding="cl100k_base"):

    # Set up the tokenizer
    encoding = tiktoken.get_encoding(embedding_encoding)

    # Generate the tokens and embeddings
    tokens = len(encoding.encode(obj))
    emb = embedding_client.get_embedding(obj)

    return tokens, emb

def write_to_csv(csv_writer, output_prefix, objectives, embedding_client):
    n = 0
    for obj in objectives:
        obj_clean = re.sub(r'^\d+\.', '', obj).strip().lstrip('- ')
        remove_words = ['Summary', 'Learning', 'Objective', 'Guiding', 'Additional', 'Question']
        if len([word for word in remove_words if word in obj_clean]) < 2:
            n += 1
            tokens, emb = generate_embedding(obj, embedding_client)
            csv_writer.writerow([output_prefix,obj_clean,tokens,emb])
    print(f"Wrote {n} learning objectives to file for {output_prefix}")

def main(input_path, embedding_client, chat_model="google/gemini-3-flash-preview", api_key=None):

    path = Path(input_path)
    output_prefix = path.stem
    output_file = output_prefix + "_learning_objectives.csv"

    if path.is_file():
        pdf_files = [input_path]
    elif path.is_dir():
        pdf_files = list(path.glob('*.pdf'))
    else:
        print("The provided path is not a valid file or directory.")
        sys.exit(1)

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['name', 'learning_objective','tokens','emb'])

        for pdf_file in pdf_files:
            objectives = define_objectives_from_pdf(pdf_file, model=chat_model, api_key=api_key)
            tag = Path(pdf_file).stem
            write_to_csv(csv_writer, tag, objectives, embedding_client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate learning objectives from PDF files')
    parser.add_argument('input_path', type=str,
                        help='Path to PDF file or directory containing PDF files')
    parser.add_argument('--provider', type=str, default='openrouter',
                        choices=['openrouter', 'ollama'],
                        help='Embedding provider: openrouter or ollama (default: openrouter)')
    parser.add_argument('--model', type=str, default='text-embedding-ada-002',
                        help='Embedding model name (default: text-embedding-ada-002)')
    parser.add_argument('--chat-model', type=str, default='google/gemini-3-flash-preview',
                        help='Chat model for generating questions (default: google/gemini-3-flash-preview)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for OpenRouter (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--ollama-url', type=str, default=None,
                        help='Ollama base URL (default: http://localhost:11434)')
    
    args = parser.parse_args()
    
    # Use provided API key or environment variable for chat (OpenRouter)
    chat_api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    
    # Validate OpenRouter API key is available for chat (will exit if not found)
    # This is needed regardless of embedding provider since chat always uses OpenRouter
    get_openrouter_api_key(api_key=chat_api_key)
    
    # Initialize embedding client
    # Only pass API key if using OpenRouter for embeddings
    embedding_api_key = chat_api_key if args.provider == 'openrouter' else None
    try:
        embedding_client = EmbeddingClient(
            provider=args.provider,
            model=args.model,
            api_key=embedding_api_key,
            base_url=args.ollama_url
        )
    except (ImportError, ValueError) as e:
        print(f"Error initializing embedding client: {e}")
        sys.exit(1)
    
    main(args.input_path, embedding_client, chat_model=args.chat_model, api_key=chat_api_key)
