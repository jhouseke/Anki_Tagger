# Anki_Tagger

This is a project I started as a fourth year medical student. Anki was a huge part of my medical education, and this felt like a productive way to give back for everything others have done.
 
Big picture, this set of scripts will parse a lecture guide and identify the most relevant anki cards within a premade Anki deck. 

While there are some fantastic deck’s out there to support medical education, aligning the content of these decks with preclinical curriculum is a persistent challenge and source of anxiety for students. This project aims to alleviate that stress by selecting the best cards for each lecture to study alongside their preclinical curriculum. Using the OpenAI API, I was able to quickly tag over 200 lectures to cover the entirety of M1 and M2 with minimal human intervention. While it is not the perfect solution, hopefully others can use and improve upon the code and make their own class tags!

# Setup

## Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

Alternatively, if you prefer using pip:
```bash
pip install -e .
```

## Configuration

All scripts read shared settings from **`config.yaml`** in the project root. Edit this file instead of passing options on the command line. You can still override any option with CLI arguments when you run a script.

- **Embedding**: provider, model, Ollama URL, token limits (used by `embed_anki_deck.py` and `make_learning_objectives.py`; keep the same so embedding dimensions match).
- **Chat**: model and token limits for generating objectives and rating cards.
- **Paths**: default paths for deck input/output, embeddings CSV, objectives CSV, cards CSV, and .apkg file. Set these in `config.yaml` to run scripts without arguments.
- **select_cards**: `max_poor_match_run`, `max_tokens_per_obj`.
- **tag_deck**: relevance score cutoffs (high / medium / minimal).

Override config file location with the `CONFIG_PATH` environment variable.

## Embedding Provider Configuration

- **OpenRouter (default)**: Set `export OPENROUTER_API_KEY=your-api-key`. Configure `embedding.provider` and `embedding.model` in `config.yaml`.
- **Ollama (local)**: Install and run Ollama; set `embedding.provider: ollama` and `embedding.model` (e.g. `nomic-embed-text`) in `config.yaml`. Optional: `embedding.ollama_url`.

# Workflow
1. Copy or edit `config.yaml` and set paths/models as needed. Set `OPENROUTER_API_KEY` if using OpenRouter.
2. In Anki, export the deck as an .apkg and as Notes (plain text) with GUID: e.g. `anki.txt`.
3. **Embed deck**: `python embed_anki_deck.py`  
   - Uses `paths.deck_input` and `paths.deck_output_prefix` from config (or `--input` / `--output`).  
   - Produces `{output_prefix}_embeddings.csv`.
4. **Learning objectives**: `python make_learning_objectives.py [path_to_pdf_or_folder]`  
   - Path can be set as `objectives.input_path` in config or passed as the argument.  
   - Produces `{stem}_learning_objectives.csv`.
5. **Select cards**: `python select_cards.py [embeddings.csv] [objectives.csv]`  
   - Paths from config if set, or pass as arguments.  
   - Produces `{prefix}_cards.csv`.
6. **Tag deck**: `python tag_deck.py [cards.csv] [deck.apkg]`  
   - Paths from config if set, or pass as arguments.  
   - Overwrites the .apkg with tagged deck.
7. Import the updated .apkg into Anki.

zachalmers

