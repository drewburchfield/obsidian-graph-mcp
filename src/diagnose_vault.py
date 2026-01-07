"""Diagnose vault files to find empty string issues."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from src.embedder import VoyageEmbedder
from src.exceptions import EmbeddingError

vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "/vault"))
embedder = VoyageEmbedder()

print(f"Scanning vault: {vault_path}")
md_files = list(vault_path.rglob("*.md"))
print(f"Found {len(md_files)} markdown files\n")

problematic_files = []
empty_files = []
successful = 0

for i, file_path in enumerate(md_files, 1):
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()

        rel_path = str(file_path.relative_to(vault_path))

        if not content:
            empty_files.append(rel_path)
            print(f"[{i}/{len(md_files)}] EMPTY: {rel_path}")
            continue

        if len(content) < 5:
            print(f"[{i}/{len(md_files)}] VERY SHORT ({len(content)} chars): {rel_path}")

        # Try to embed single file
        try:
            embedding = embedder.embed(content, input_type="document")
            successful += 1
            if i % 10 == 0:
                print(f"[{i}/{len(md_files)}] ✓ Progress: {successful} successful")

        except EmbeddingError as e:
            problematic_files.append((rel_path, f"Embedding failed: {e}"))
            print(f"[{i}/{len(md_files)}] FAILED: {rel_path} - {e}")

    except Exception as e:
        problematic_files.append((rel_path, str(e)))
        print(f"[{i}/{len(md_files)}] ERROR: {rel_path} - {e}")

print(f"\n{'='*60}")
print("Summary:")
print(f"  Successful: {successful}/{len(md_files)}")
print(f"  Empty files: {len(empty_files)}")
print(f"  Problematic: {len(problematic_files)}")

if empty_files:
    print(f"\nEmpty files ({len(empty_files)}):")
    for path in empty_files[:10]:
        print(f"  - {path}")
    if len(empty_files) > 10:
        print(f"  ... and {len(empty_files) - 10} more")

if problematic_files:
    print(f"\nProblematic files ({len(problematic_files)}):")
    for path, error in problematic_files[:10]:
        print(f"  - {path}: {error}")
