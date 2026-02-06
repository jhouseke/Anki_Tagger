"""
Run the full Anki_Tagger workflow from a single entrypoint.

All scripts use default filenames in the project root (see config_loader).
Stages: embed_anki_deck → make_learning_objectives → select_cards → tag_deck.
Before each stage, required input files are checked; missing files raise a clear error.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from config_loader import (
    load_config,
    PROJECT_ROOT,
    DECK_EXPORT,
    EMBEDDINGS_CSV,
    LEARNING_OBJECTIVES_CSV,
    CARDS_CSV,
    DECK_APKG,
    path_in_project,
)


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    cfg = load_config()
    obj_cfg = cfg.get("objectives", {})

    parser = argparse.ArgumentParser(description="Run full Anki_Tagger pipeline")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embed_anki_deck stage")
    parser.add_argument("--skip-objectives", action="store_true", help="Skip make_learning_objectives stage")
    parser.add_argument("--skip-select", action="store_true", help="Skip select_cards stage")
    parser.add_argument("--skip-tag", action="store_true", help="Skip tag_deck stage")
    args = parser.parse_args()

    # (stage_name, script_name, list of paths that must exist before running)
    stages = []
    if not args.skip_embed:
        stages.append(("Embed deck", "embed_anki_deck.py", [path_in_project(DECK_EXPORT)]))
    if not args.skip_objectives:
        stages.append(("Make objectives", "make_learning_objectives.py", None))  # script checks input_path
    if not args.skip_select:
        stages.append(("Select cards", "select_cards.py", [path_in_project(EMBEDDINGS_CSV), path_in_project(LEARNING_OBJECTIVES_CSV)]))
    if not args.skip_tag:
        stages.append(("Tag deck", "tag_deck.py", [path_in_project(CARDS_CSV), path_in_project(DECK_APKG)]))

    def check_required(paths_to_check):
        for p in (paths_to_check or []):
            if not Path(p).exists():
                raise SystemExit(f"Missing required file for this stage: {p}")

    if not args.skip_objectives and not obj_cfg.get("input_path"):
        raise SystemExit(
            "make_learning_objectives needs a PDF or directory. Set objectives.input_path in config.yaml."
        )

    with tqdm(total=len(stages), desc="Pipeline", unit="stage", dynamic_ncols=True) as pbar:
        for name, script, required_files in stages:
            check_required(required_files)
            print(f"\n== {name} ==")
            cmd = [sys.executable, str(PROJECT_ROOT / script)]
            _run(cmd, cwd=str(PROJECT_ROOT))
            pbar.update(1)


if __name__ == "__main__":
    main()

