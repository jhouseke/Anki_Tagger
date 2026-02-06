import os, re, sys, csv, glob, time, argparse
import tiktoken
import pdfplumber
from embedding_client import EmbeddingClient, EmbeddingProvider
from pathlib import Path
from config_loader import load_config, get_api_key, PROJECT_ROOT, LEARNING_OBJECTIVES_CSV
from tqdm import tqdm

try:
    from openrouter import OpenRouter
except ImportError:
    OpenRouter = None

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

def generate_questions(prompt, model="google/gemini-3-flash-preview", temperature=1.0, api_key=None, max_tokens=32000, token_buffer=1000):
    formatted_prompt = [{"role": "system", "content": "You are a socratic medical school tutor that provides comprehensive learning questions."},
                        {"role": "user", "content": prompt}]
    remaining_tokens = max_tokens - count_tokens(" ".join([message["content"] for message in formatted_prompt])) - token_buffer

    if remaining_tokens < token_buffer:
        print(f"Warning! Input text is longer than the model can support. Consider trimming input and trying again.")
        print(f"Current length: {count_tokens(prompt)}, recommended < {max_tokens - token_buffer}")
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
def define_objectives_from_pdf(pdf_file, model="google/gemini-3-flash-preview", temperature=1.0, api_key=None, max_tokens=32000, token_buffer=1000):
    text = extract_text_from_pdf(pdf_file)
    prompt = f"Generate a list of learning questions that comprehensively covers the most important information presented in the text below to understand the topics presented.\n\n{text}"
    generated_text = generate_questions(prompt, model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens, token_buffer=token_buffer)
    objectives = [line.strip() for line in generated_text.split("\n") if line.strip()]
    return objectives

@handle_api_error
def generate_embedding(obj, embedding_client, embedding_encoding="cl100k_base"):
    encoding = tiktoken.get_encoding(embedding_encoding)

    # Generate the tokens and embeddings
    tokens = len(encoding.encode(obj))
    emb = embedding_client.get_embedding(obj)

    return tokens, emb

def write_to_csv(csv_writer, output_prefix, objectives, embedding_client, embedding_encoding="cl100k_base"):
    n = 0
    for obj in objectives:
        obj_clean = re.sub(r'^\d+\.', '', obj).strip().lstrip('- ')
        remove_words = ['Summary', 'Learning', 'Objective', 'Guiding', 'Additional', 'Question']
        if len([word for word in remove_words if word in obj_clean]) < 2:
            n += 1
            tokens, emb = generate_embedding(obj, embedding_client, embedding_encoding=embedding_encoding)
            csv_writer.writerow([output_prefix,obj_clean,tokens,emb])
    print(f"Wrote {n} learning objectives to file for {output_prefix}")

def main(input_path, embedding_client, chat_model="google/gemini-3-flash-preview", api_key=None, max_tokens=32000, token_buffer=1000, embedding_encoding="cl100k_base"):

    path = Path(input_path)
    output_file = PROJECT_ROOT / LEARNING_OBJECTIVES_CSV

    if path.is_file():
        pdf_files = [input_path]
    elif path.is_dir():
        pdf_files = list(path.glob('*.pdf'))
    else:
        print("The provided path is not a valid file or directory.")
        sys.exit(1)

    file_exists = os.path.exists(output_file)
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Only write header on first creation (prevents duplicated header rows)
        if (not file_exists) or os.path.getsize(output_file) == 0:
            csv_writer.writerow(['name', 'learning_objective','tokens','emb'])

        for pdf_file in tqdm(pdf_files, desc="PDFs", unit="pdf", dynamic_ncols=True):
            objectives = define_objectives_from_pdf(pdf_file, model=chat_model, api_key=api_key, max_tokens=max_tokens, token_buffer=token_buffer)
            tag = Path(pdf_file).stem
            write_to_csv(csv_writer, tag, objectives, embedding_client, embedding_encoding=embedding_encoding)

if __name__ == "__main__":
    cfg = load_config()
    emb = cfg["embedding"]
    chat = cfg["chat"]
    obj_cfg = cfg.get("objectives", {})

    parser = argparse.ArgumentParser(description='Generate learning objectives from PDF files')
    parser.add_argument('input_path', type=str, nargs='?', default=obj_cfg.get("input_path"),
                        help='Path to PDF file or directory (default from config)')
    parser.add_argument('--provider', type=str, default=emb.get("provider", "openrouter"),
                        choices=['openrouter', 'ollama'],
                        help='Embedding provider (default from config)')
    parser.add_argument('--model', type=str, default=emb.get("model"),
                        help='Embedding model (default from config)')
    parser.add_argument('--chat-model', type=str, default=chat.get("model"),
                        help='Chat model for questions (default from config)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenRouter API key (or OPENROUTER_API_KEY env)')
    parser.add_argument('--ollama-url', type=str, default=emb.get("ollama_url"),
                        help='Ollama base URL (default from config)')
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path:
        parser.error("input_path is required: set objectives.input_path in config.yaml or pass as argument")
    chat_api_key = args.api_key or get_api_key(cfg)
    get_openrouter_api_key(api_key=chat_api_key)
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

    main(
        input_path,
        embedding_client,
        chat_model=args.chat_model,
        api_key=chat_api_key,
        max_tokens=chat.get("max_tokens", 32000),
        token_buffer=chat.get("token_buffer", 1000),
        embedding_encoding=emb.get("encoding", "cl100k_base"),
    )
