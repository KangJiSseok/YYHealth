import os
from pathlib import Path
from typing import Dict, List
import json

from dotenv import load_dotenv

from app.core.pdf_chunker import process_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"

# 특정 PDF별로 처리할 페이지 범위를 지정 (start, end 포함)
PAGE_OVERRIDES = {
    "2020korean.pdf": [(40, 66), (82, 106), (118, 134), (146, 187), (202, 231)],
}


def run_all_pdfs() -> List[Dict]:
    """
    data/pdfs 경로에 있는 모든 PDF를 순회하며
    process_pdf를 실행하고 모든 chunk를 하나의 리스트로 합친다.
    """
    all_chunks: List[Dict] = []

    if not PDF_DIR.is_dir():
        raise RuntimeError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"[runner] No PDF files found in {PDF_DIR}")
        return all_chunks

    print(f"[runner] Found {len(pdf_files)} PDF files")

    for filename in pdf_files:
        pdf_path = PDF_DIR / filename

        print("=" * 70)
        print(f"[runner] Processing: {pdf_path}")

        try:
            page_ranges = PAGE_OVERRIDES.get(filename)
            chunks = process_pdf(pdf_path, page_ranges=page_ranges)
            all_chunks.extend(chunks)
            print(f"[runner] {filename}: {len(chunks)} chunks")
        except Exception as e:
            print(f"[runner][ERROR] Failed to process {filename}")
            print(e)

    print("=" * 70)
    print(f"[runner] TOTAL chunks generated: {len(all_chunks)}")

    return all_chunks


if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")
    chunks = run_all_pdfs()
    if chunks:
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CHUNKS_PATH.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"[runner] Saved {len(chunks)} chunks to {CHUNKS_PATH}")
    print(f"[runner] TOTAL chunks generated:")
