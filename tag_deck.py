import os, sys
import shutil
import zipfile
import subprocess
import pandas as pd
from anki.collection import Collection

HIGH_RELEVANCE_CUTOFF = 70
MEDIUM_RELEVANCE_CUTOFF = 40
REMOVE_RELEVANCE_CUTOFF = 10

def main(card_path, anki_apkg):

    # Load the csv file into a DataFrame
    df = pd.read_csv(card_path)
    df = df.fillna(0)

    # Group by 'guid' and keep only the row with the highest 'score' for each group
    df = df.loc[df.groupby('guid')['score'].idxmax()]

    # Unzip the .apkg file to a temporary folder
    with zipfile.ZipFile(anki_apkg, 'r') as zip_ref:
        zip_ref.extractall("temp_folder")

    # The main .anki2 database file should have been extracted now
    # Look for the .anki21 or .anki21b file in the temp_folder
    anki2_files = [f for f in os.listdir("temp_folder") if f.endswith(".anki21") or f.endswith(".anki21b") or f.endswith(".anki2")]
    
    if not anki2_files:
        raise FileNotFoundError("No Anki database file (.anki2, .anki21, or .anki21b) found in temp_folder")
    
    # Categorize files by type
    anki21b_files = [f for f in anki2_files if f.endswith(".anki21b")]
    anki21_files = [f for f in anki2_files if f.endswith(".anki21")]
    anki2_old_files = [f for f in anki2_files if f.endswith(".anki2")]
    
    # Try to open all files and use the one with the most notes
    anki2_file = None
    max_notes = 0
    
    # Try all .anki2 files
    for test_file in anki2_old_files:
        try:
            test_path = os.path.join("temp_folder", test_file)
            test_col = Collection(test_path)
            test_notes = test_col.db.all("SELECT guid FROM notes")
            note_count = len(test_notes)
            test_col.close()
            if note_count > max_notes:
                max_notes = note_count
                anki2_file = test_file
        except Exception:
            pass
    
    # Try all .anki21 files
    for test_file in anki21_files:
        try:
            test_path = os.path.join("temp_folder", test_file)
            test_col = Collection(test_path)
            test_notes = test_col.db.all("SELECT guid FROM notes")
            note_count = len(test_notes)
            test_col.close()
            if note_count > max_notes:
                max_notes = note_count
                anki2_file = test_file
        except Exception:
            pass
    
    # Try all .anki21b files (these are Zstandard compressed - need to decompress first)
    for test_file in anki21b_files:
        try:
            test_path = os.path.join("temp_folder", test_file)
            decompressed_path = os.path.join("temp_folder", test_file.replace(".anki21b", "_decompressed.db"))
            
            # Decompress the zstd-compressed file
            result = subprocess.run(["zstd", "-d", test_path, "-o", decompressed_path, "-f"], 
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"zstd decompression failed: {result.stderr}")
            
            test_col = Collection(decompressed_path)
            test_notes = test_col.db.all("SELECT guid FROM notes")
            note_count = len(test_notes)
            test_col.close()
            
            if note_count > max_notes:
                max_notes = note_count
                anki2_file = decompressed_path  # Use the decompressed file path
        except Exception:
            pass
    
    if not anki2_file:
        raise FileNotFoundError("No valid Anki database file could be opened. Tried: " + ", ".join(anki2_files))
    
    # Initialize a new collection with the working file
    # If anki2_file is already a full path (from decompressed .anki21b), use it directly
    if os.path.dirname(anki2_file):
        col_path = anki2_file
    else:
        col_path = os.path.join("temp_folder", anki2_file)
    col = Collection(col_path)
    tagged = set()

    # Get all notes from database
    all_notes_query = col.db.all("SELECT guid FROM notes")
    total_notes_count = len(all_notes_query)
    
    # Warn if database has very few notes compared to CSV
    csv_guid_count = len(df['guid'].unique())
    if total_notes_count < csv_guid_count * 0.1:  # Less than 10% of CSV GUIDs
        print(f"\n⚠️  WARNING: Database mismatch detected!")
        print(f"   Database has {total_notes_count} note(s)")
        print(f"   CSV has {csv_guid_count} unique GUID(s)")
        print(f"   This suggests the .apkg file is from a different deck than the one used to generate the CSV.")
        print(f"   Please ensure you are using the correct .apkg file that corresponds to your CSV\n")
    
    # Check GUID matches
    csv_guids_set = set(df['guid'].astype(str).str.strip())
    db_guids = set([g[0] for g in all_notes_query])
    matching_guids = csv_guids_set.intersection(db_guids)
    
    print(f"Total GUIDs in CSV: {len(csv_guids_set)}")
    print(f"Total GUIDs in database: {len(db_guids)}")
    print(f"Matching GUIDs: {len(matching_guids)}")
    
    if len(matching_guids) == 0:
        print("WARNING: No GUIDs match between CSV and database!")
        print("This suggests the .apkg file may be from a different deck than the one used to create embeddings.")

    # Iterate through all cards and apply tags
    for index, row in df.iterrows():

        guid = str(row['guid']).strip()
        tag = row['tag']
        score = int(row['score'])

        if score >= HIGH_RELEVANCE_CUTOFF:
            try:
                result = col.db.all("SELECT id, tags FROM notes WHERE guid = ?", guid)
                if result:
                    note_id, note_tags = result[0]
                    new_tag = note_tags + " " + tag+"::1_highly_relevant" + " "
                    col.db.execute("UPDATE notes set tags = ? where id = ?", new_tag, note_id)
                    tagged.add(guid)
            except Exception as e:
                print(f"Error processing guid {guid}: {e}")

        if score < HIGH_RELEVANCE_CUTOFF and score >= MEDIUM_RELEVANCE_CUTOFF:
            try:
                result = col.db.all("SELECT id, tags FROM notes WHERE guid = ?", guid)
                if result:
                    note_id, note_tags = result[0]
                    new_tag = note_tags + " " + tag+"::2_somewhat_relevant" + " "
                    col.db.execute("UPDATE notes set tags = ? where id = ?", new_tag, note_id)
                    tagged.add(guid)
            except Exception as e:
                print(f"Error processing guid {guid}: {e}")

        if score < MEDIUM_RELEVANCE_CUTOFF and score >= REMOVE_RELEVANCE_CUTOFF:
            try:
                result = col.db.all("SELECT id, tags FROM notes WHERE guid = ?", guid)
                if result:
                    note_id, note_tags = result[0]
                    new_tag = note_tags + " " + tag+"::3_minimally_relevant" + " "
                    col.db.execute("UPDATE notes set tags = ? where id = ?", new_tag, note_id)
                    tagged.add(guid)
            except Exception as e:
                print(f"Error processing guid {guid}: {e}")

    # Checkpoint WAL to merge changes into main database before closing
    col.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    
    # Save the collection
    col.close()

    # If we used a decompressed .anki21b file, recompress it back
    if "_decompressed.db" in col_path:
        # Remove any WAL/SHM files that SQLite may have created
        for suffix in ["-wal", "-shm", "-journal"]:
            wal_file = col_path + suffix
            if os.path.exists(wal_file):
                os.remove(wal_file)
        
        original_anki21b = col_path.replace("_decompressed.db", ".anki21b")
        result = subprocess.run(["zstd", col_path, "-o", original_anki21b, "-f"], 
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to recompress .anki21b: {result.stderr}")
        else:
            # Remove the decompressed file
            os.remove(col_path)

    # Re-create the .apkg file (exclude temporary files)
    exclude_patterns = ["-wal", "-shm", "-journal", "_decompressed"]
    with zipfile.ZipFile(anki_apkg, 'w') as zip_ref:
        for filename in os.listdir("temp_folder"):
            # Skip temporary SQLite files
            if any(pattern in filename for pattern in exclude_patterns):
                continue
            zip_ref.write(os.path.join("temp_folder", filename), arcname=filename)

    # Clean up the temporary folder
    print(f"Tagged {len(tagged)} cards. Process Complete")
    shutil.rmtree("temp_folder")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: tag_deck.py <cards.csv> <anki_deck.apkg>")
        sys.exit(1)
    card_path = sys.argv[1]
    anki_apkg = sys.argv[2]
    main(card_path, anki_apkg)
