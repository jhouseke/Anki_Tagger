import pandas as pd
import numpy as np
import re, sys, csv, os, argparse
from pathlib import Path
import tiktoken
import time
from config_loader import (
    load_config,
    get_api_key,
    PROJECT_ROOT,
    EMBEDDINGS_CSV,
    LEARNING_OBJECTIVES_CSV,
    CARDS_CSV,
    PROGRESS_CSV,
    require_file,
)
from tqdm import tqdm

try:
    from openrouter import OpenRouter
except ImportError:
    OpenRouter = None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
        t=5
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle various API errors (rate limits, connection errors, etc.)
                error_str = str(e).lower()
                if 'rate limit' in error_str or '429' in error_str or 'quota' in error_str:
                    print(f'Rate limit exceeded. Waiting {t}s before retrying.')
                elif 'connection' in error_str or 'timeout' in error_str:
                    print(f'Connection error. Waiting {t}s before retrying.')
                else:
                    print(f'API Error: {e}. Waiting {t}s before retrying.')
                time.sleep(t)
                t += 5
    return wrapper

def convert_to_np_array(s):
    return np.fromstring(s.strip("[]"), sep=",")

def load_emb(path):

    # Be tolerant of duplicate header rows / bad values (e.g. 'tokens' as a value)
    # and support both embeddings CSV (guid, card, tokens, emb) and objectives CSV
    # (name, learning_objective, tokens, emb).
    df = pd.read_csv(
        path,
        dtype={"guid": str, "card": str, "name": str, "learning_objective": str},
        converters={"emb": convert_to_np_array},
    )

    if "tokens" in df.columns:
        df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
        df = df.dropna(subset=["tokens"]).astype({"tokens": int})

    return df

def vs(x, y):
    return np.dot(np.array(x), np.array(y))

def construct_prompt(obj,card):

    prompt = f"Rate how relevant the anki card is to the learning question on a scale from 0 to 100 and return the score.\n\
    Learning question: {obj}\n\
    Anki card: {card}"

    formatted_prompt = [{"role": "system", "content": "You are an assistant that precisely follows instructions."},
                        {"role": "user", "content": prompt}]

    return formatted_prompt

def count_tokens(text):
    # Use cl100k_base encoding as a general estimate (works reasonably well for most models)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = list(enc.encode(text))
    return len(tokens)

def tokens_in_prompt(formatted_prompt):
    formatted_prompt_str = ""
    for message in formatted_prompt:
        formatted_prompt_str += message["content"] + " "
    return count_tokens(formatted_prompt_str)

@handle_api_error
def rate_card_for_obj(prompt, model="google/gemini-3-flash-preview", temperature=1, api_key=None, max_tokens=32000, token_buffer=1000):

    prompt_tokens = tokens_in_prompt(prompt)
    remaining_tokens = max_tokens - prompt_tokens - token_buffer

    if remaining_tokens < token_buffer:
        print(f"Warning! Input text is longer than the model can support. Consider trimming input.")
        remaining_tokens = token_buffer

    # OpenRouter client must be used as a context manager
    api_key = get_openrouter_api_key(api_key=api_key)
    with OpenRouter(api_key=api_key) as openrouter:
        response = openrouter.chat.send(
            model=model,
            messages=prompt,
            max_tokens=remaining_tokens,
            temperature=temperature
        )
        # Access response inside context manager
        string_return = response.choices[0].message.content.strip()
    
    return string_return.replace('\n',' ')

def clean_reply(s):

    matches = re.search(r'Score: (\d{1,3})', s)

    if matches:
        score = matches.group(1)
        return int(score)

    else:
        matches = re.findall(r'\b([0-9][0-9]?|100)\b', s)
        if matches:
            numbers = [int(num) for num in matches]
            return min(numbers)
        else:
            return "NA"

def _first_vector_len(series):
    """Return length of first non-empty embedding vector in a Series, else None."""
    for v in series:
        if v is None:
            continue
        arr = np.array(v)
        if arr.size:
            return int(arr.size)
    return None


def main(emb_path, obj_path, chat_model="google/gemini-3-flash-preview", api_key=None, max_tokens=32000, token_buffer=1000, max_poor_match_run=10, max_tokens_per_obj=5000):

    emb_path = str(Path(emb_path).resolve())
    obj_path = str(Path(obj_path).resolve())
    cards_file = PROJECT_ROOT / CARDS_CSV
    progress_file = PROJECT_ROOT / PROGRESS_CSV

    if cards_file.exists():
        print(f"Cards already selected: {cards_file} (skipping)")
        return

    # load previous progress if exists
    last_processed_index = -1
    if progress_file.exists():
        last_progress_df = pd.read_csv(progress_file)
        if not last_progress_df.empty:
            last_processed_index = int(last_progress_df.iloc[-1][0])

    emb_df = load_emb(emb_path)
    obj_df = load_emb(obj_path)

    # Hard fail early if embedding dimensions don't match (common source of silent bad results)
    deck_dim = _first_vector_len(emb_df.get("emb", []))
    obj_dim = _first_vector_len(obj_df.get("emb", []))
    if deck_dim and obj_dim and deck_dim != obj_dim:
        raise ValueError(
            f"Embedding dimension mismatch: deck embeddings are {deck_dim}D but objectives are {obj_dim}D.\n"
            "Fix: ensure `embedding.provider` and `embedding.model` are the SAME when running\n"
            "`embed_anki_deck.py` and `make_learning_objectives.py`, then regenerate both CSVs."
        )

    with open(cards_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if last_processed_index == -1:  # if there's no previous progress
            csv_writer.writerow(['guid','card','tag','cosine_sim','gpt_reply','score','objective'])

        for obj_index, obj_row in tqdm(list(obj_df.iterrows()), desc="Objectives", unit="obj", dynamic_ncols=True):

            if obj_index <= last_processed_index:
                continue  # skip if the row has already been processed

            print(f"Processing objective {obj_index}")
            tag = obj_row['name']
            obj = obj_row['learning_objective']
            tokens = obj_row['tokens']
            obj_emb = obj_row['emb']

            # Vectorized similarity would be faster, but keep simple + show progress.
            emb_df["cosine_sim"] = emb_df.emb.apply(lambda x: vs(obj_emb, x))
            emb_df.sort_values(by='cosine_sim', ascending=False, inplace=True)

            poor_match_run_count = 0
            tokens_used = 0

            for index, emb_row in tqdm(list(emb_df.iterrows()), desc="Cards", unit="card", leave=False, dynamic_ncols=True):

                if poor_match_run_count > max_poor_match_run or tokens_used > max_tokens_per_obj:
                    break

                guid = emb_row['guid']
                card = emb_row['card']
                cosine_sim = emb_row["cosine_sim"]
                gpt_reply = "NA"
                score = "NA"

                prompt = construct_prompt(obj,card)
                tokens_used += tokens_in_prompt(prompt)

                #try with progressively more creative juice
                temp = 0
                while score == "NA" and temp <= 1:
                    gpt_reply = rate_card_for_obj(prompt, model=chat_model, temperature=temp, api_key=api_key, max_tokens=max_tokens, token_buffer=token_buffer)
                    score = clean_reply(gpt_reply)
                    temp += 0.25

                csv_writer.writerow([guid,card,tag,cosine_sim,gpt_reply,score,obj])
                if score > 50:
                    poor_match_run_count=0
                else:
                    poor_match_run_count+=1

            with open(progress_file, 'a', newline='', encoding='utf-8') as progress_csvfile:
                progress_csv_writer = csv.writer(progress_csvfile)
                progress_csv_writer.writerow([obj_index])

if __name__ == "__main__":
    cfg = load_config()
    chat = cfg.get("chat", {})
    sc = cfg.get("select_cards", {})

    parser = argparse.ArgumentParser(description='Select Anki cards matching learning objectives')
    parser.add_argument('emb_path', type=str, nargs='?', default=None,
                        help=f'Deck embeddings CSV (default: project root / {EMBEDDINGS_CSV})')
    parser.add_argument('obj_path', type=str, nargs='?', default=None,
                        help=f'Learning objectives CSV (default: project root / {LEARNING_OBJECTIVES_CSV})')
    parser.add_argument('--chat-model', type=str, default=chat.get("model"),
                        help='Chat model for rating (default from config)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenRouter API key (or OPENROUTER_API_KEY env)')
    args = parser.parse_args()

    if args.emb_path and args.obj_path:
        emb_path = args.emb_path
        obj_path = args.obj_path
    else:
        emb_path = str(require_file(EMBEDDINGS_CSV, "select_cards (embeddings)"))
        obj_path = str(require_file(LEARNING_OBJECTIVES_CSV, "select_cards (objectives)"))
    api_key = args.api_key or get_api_key(cfg)
    get_openrouter_api_key(api_key=api_key)

    main(
        emb_path,
        obj_path,
        chat_model=args.chat_model,
        api_key=api_key,
        max_tokens=chat.get("max_tokens", 32000),
        token_buffer=chat.get("token_buffer", 1000),
        max_poor_match_run=sc.get("max_poor_match_run", 10),
        max_tokens_per_obj=sc.get("max_tokens_per_obj", 5000),
    )
