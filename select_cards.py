import pandas as pd
import numpy as np
import re, sys, csv, os, argparse
import tiktoken
import time

try:
    from openrouter import OpenRouter
except ImportError:
    OpenRouter = None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

MAX_POOR_MATCH_RUN = 10
MAX_TOKENS_PER_OBJ = 5000
MAX_TOKENS = 32000  # Increased for modern models
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

    # Specify the data types for columns 0, 1, and 2
    column_dtypes = {0: str, 1: str, 2: int}

    # Read CSV file and interpret column types
    df = pd.read_csv(
        path,
        dtype=column_dtypes,
        converters={3: convert_to_np_array})

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
def rate_card_for_obj(prompt, model="google/gemini-3-flash-preview", temperature=1, api_key=None):

    # Calculate the remaining tokens for the response
    prompt_tokens = tokens_in_prompt(prompt)
    remaining_tokens = MAX_TOKENS - prompt_tokens - TOKEN_BUFFER

    if remaining_tokens < TOKEN_BUFFER:
        print(f"Warning! Input text is longer than the model can support. Consider trimming input.")
        remaining_tokens = TOKEN_BUFFER

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

def main(emb_path, obj_path, chat_model="google/gemini-3-flash-preview", api_key=None):

    output_prefix = os.path.basename(obj_path).replace("_learning_objectives.csv",'')

    # load previous progress if exists
    last_processed_index = -1
    progress_file = f"{output_prefix}_progress.csv"

    if os.path.exists(progress_file):
        last_progress_df = pd.read_csv(progress_file)
        if not last_progress_df.empty:
            last_processed_index = last_progress_df.iloc[-1][0]

    emb_df = load_emb(emb_path)
    obj_df = load_emb(obj_path)

    with open(f'{output_prefix}_cards.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if last_processed_index == -1:  # if there's no previous progress
            csv_writer.writerow(['guid','card','tag','cosine_sim','gpt_reply','score','objective'])

        for obj_index,obj_row in obj_df.iterrows():

            if obj_index <= last_processed_index:
                continue  # skip if the row has already been processed

            print(f"Processing objective {obj_index}")
            tag = obj_row['name']
            obj = obj_row['learning_objective']
            tokens = obj_row['tokens']
            obj_emb = obj_row['emb']

            emb_df["cosine_sim"] = emb_df.emb.apply(lambda x: vs(obj_emb,x))
            emb_df.sort_values(by='cosine_sim', ascending=False, inplace=True)

            poor_match_run_count = 0
            tokens_used = 0

            for index,emb_row in emb_df.iterrows():

                if poor_match_run_count > MAX_POOR_MATCH_RUN or tokens_used > MAX_TOKENS_PER_OBJ:
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
                    gpt_reply = rate_card_for_obj(prompt, model=chat_model, temperature=temp, api_key=api_key)
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
    parser = argparse.ArgumentParser(description='Select Anki cards matching learning objectives')
    parser.add_argument('emb_path', type=str,
                        help='Path to deck embeddings CSV file')
    parser.add_argument('obj_path', type=str,
                        help='Path to learning objectives CSV file')
    parser.add_argument('--chat-model', type=str, default='google/gemini-3-flash-preview',
                        help='Chat model for rating cards (default: google/gemini-3-flash-preview)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for OpenRouter (or set OPENROUTER_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Use provided API key or environment variable
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    
    # Validate OpenRouter API key is available (will exit if not found)
    get_openrouter_api_key(api_key=api_key)
    
    main(args.emb_path, args.obj_path, chat_model=args.chat_model, api_key=api_key)
