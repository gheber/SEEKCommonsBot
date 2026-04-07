#!/usr/bin/env python3
"""
SEEK records -> HDF5 vector DB with record metadata + query

Usage:
  # 1) Build once (defaults to sample.txt)
  python3 seek_vectors.py build --out seek.h5

  # Build from a specific input file
  python3 seek_vectors.py build --input sample.txt --out seek.h5

  # Force rebuild
  python3 seek_vectors.py build --out seek.h5 --rebuild

  # 2) List indexed records
  python3 seek_vectors.py list-works --h5 seek.h5

  # 3) Query globally
  python3 seek_vectors.py query --h5 seek.h5 --text "arabic language" --top-k 10

  # 4) Query within a specific record (QID or label substring match)
  python3 seek_vectors.py query --h5 seek.h5 --text "lingua franca" --work "Q13955" --top-k 5
"""

import argparse
import datetime as dt
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


TOKENIZER_NAME = "bert-base-uncased"
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ID_RE = re.compile(r"^id\s+(Q\d+)\s*$")


def read_input_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Missing dependency: h5py") from exc
    return h5py


def require_embedding_dependencies():
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for build/query: sentence-transformers and transformers"
        ) from exc
    return SentenceTransformer, AutoTokenizer


def parse_records(text: str) -> List[Dict[str, str]]:
    """
    Parse records that begin with 'id Q...' and continue until the next record.

    Some records contain internal blank lines, so parsing by QID header is more
    reliable than splitting on empty lines alone.
    """
    records: List[Dict[str, str]] = []
    current_lines: List[str] = []
    current_qid: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = ID_RE.match(line.strip())
        if match:
            if current_qid is not None:
                records.append(build_record(current_qid, current_lines))
            current_qid = match.group(1)
            current_lines = [line.strip()]
            continue

        if current_qid is None:
            if line.strip():
                raise ValueError(f"Found content before first record header: {line!r}")
            continue

        current_lines.append(line)

    if current_qid is not None:
        records.append(build_record(current_qid, current_lines))

    if not records:
        raise ValueError("No records found. Expected lines starting with 'id Q...'.")

    return records


def build_record(qid: str, lines: List[str]) -> Dict[str, str]:
    label = ""
    description = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("Label ") and not label:
            label = stripped[len("Label ") :].strip()
        elif stripped.startswith("Description ") and not description:
            description = stripped[len("Description ") :].strip()

    title = f"{qid} | {label}" if label else qid
    text = "\n".join(lines).strip()
    return {
        "qid": qid,
        "label": label,
        "description": description,
        "title": title,
        "text": text,
    }


def chunk_by_tokens(
    tokenizer: Any,
    text: str,
    max_tokens: int = 256,
    overlap: int = 32,
) -> List[str]:
    """
    Split text into chunks by tokenizer length; return decoded strings.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    chunks = []
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    for start in range(0, len(ids), step):
        window = ids[start : start + max_tokens]
        if not window:
            break
        chunk_text = tokenizer.decode(window, skip_special_tokens=True)
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def get_embedder(name: str = EMBEDDER_NAME):
    SentenceTransformer, _ = require_embedding_dependencies()
    return SentenceTransformer(name)


def embed_texts(
    embedder: Any, texts: List[str], batch_size: int = 64
) -> np.ndarray:
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def write_hdf5(
    h5_path: str,
    all_chunks: List[str],
    all_work_ids: List[int],
    embeddings: np.ndarray,
    works_index: List[Dict[str, object]],
    meta: Dict[str, str],
):
    assert len(all_chunks) == embeddings.shape[0] == len(all_work_ids)

    h5py = require_h5py()
    vlen_str = h5py.string_dtype("utf-8")

    with h5py.File(h5_path, "w") as h5:
        gmeta = h5.create_group("meta")
        for key, value in meta.items():
            gmeta.attrs[key] = value

        gchunks = h5.create_group("chunks")
        gchunks.create_dataset(
            "text",
            data=np.array(all_chunks, dtype=object),
            dtype=vlen_str,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        gchunks.create_dataset(
            "work_id",
            data=np.asarray(all_work_ids, dtype=np.int32),
            dtype=np.int32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        gchunks.create_dataset(
            "embedding",
            data=embeddings,
            dtype=np.float32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )

        gworks = h5.create_group("works")
        index_dtype = np.dtype(
            [
                ("work_id", np.int32),
                ("title", vlen_str),
                ("start_row", np.int64),
                ("end_row", np.int64),
            ]
        )
        rows = np.empty(len(works_index), dtype=index_dtype)
        for i, rec in enumerate(works_index):
            rows[i] = (
                np.int32(rec["work_id"]),
                rec["title"],
                np.int64(rec["start_row"]),
                np.int64(rec["end_row"]),
            )
        gworks.create_dataset(
            "index",
            data=rows,
            dtype=index_dtype,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )


def build_if_missing(
    out_path: str,
    input_path: str = "sample.txt",
    rebuild: bool = False,
):
    if os.path.exists(out_path) and not rebuild:
        print(f"[build] HDF5 already exists at {out_path}. Skipping regeneration.")
        return

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    print(f"[build] Reading records from {input_path}...")
    text = read_input_text(input_path)

    print("[build] Parsing SEEK records...")
    records = parse_records(text)
    print(f"[build] Found {len(records)} records.")

    _, AutoTokenizer = require_embedding_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    embedder = get_embedder()

    all_chunks: List[str] = []
    all_work_ids: List[int] = []
    works_index: List[Dict[str, object]] = []
    row_cursor = 0

    for work_id, record in enumerate(records):
        chunks = chunk_by_tokens(tokenizer, record["text"], max_tokens=256, overlap=32)
        if not chunks:
            continue
        start_row = row_cursor
        all_chunks.extend(chunks)
        all_work_ids.extend([work_id] * len(chunks))
        row_cursor += len(chunks)
        works_index.append(
            {
                "work_id": work_id,
                "title": record["title"],
                "start_row": start_row,
                "end_row": row_cursor,
            }
        )

    if not all_chunks:
        raise ValueError("No text chunks were generated from the input records.")

    print(f"[build] Total chunks: {len(all_chunks)}")

    print("[build] Embedding chunks (L2-normalized)...")
    embeddings = embed_texts(embedder, all_chunks, batch_size=64)

    meta = {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "source": os.path.abspath(input_path),
        "records": str(len(records)),
        "tokenizer": TOKENIZER_NAME,
        "embedder": EMBEDDER_NAME,
        "embed_dim": str(embeddings.shape[1]),
        "chunking": "max_tokens=256,overlap=32 (BERT tokenizer)",
        "note": "Embeddings are unit-normalized; cosine ~= dot product",
    }

    print(f"[build] Writing HDF5 -> {out_path}")
    write_hdf5(
        h5_path=out_path,
        all_chunks=all_chunks,
        all_work_ids=all_work_ids,
        embeddings=embeddings,
        works_index=works_index,
        meta=meta,
    )
    print("[build] Done.")


def load_works_index(h5: Any) -> List[Dict[str, object]]:
    idx = h5["works/index"]
    out = []
    for row in idx:
        out.append(
            {
                "work_id": int(row["work_id"]),
                "title": str(row["title"]),
                "start_row": int(row["start_row"]),
                "end_row": int(row["end_row"]),
            }
        )
    return out


def find_work_rows(
    works_index: List[Dict[str, object]], title_substring: str
) -> Optional[Tuple[int, int]]:
    query = title_substring.lower()
    for rec in works_index:
        if query in rec["title"].lower():
            return (rec["start_row"], rec["end_row"])
    return None


def query_hdf5(
    h5_path: str,
    query_text: str,
    top_k: int = 5,
    work_filter: Optional[str] = None,
    model_name: str = EMBEDDER_NAME,
) -> List[Dict[str, object]]:
    """
    Returns: list of {rank, score, row, work_id, work_title, text}
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    embedder = get_embedder(model_name)
    qv = embedder.encode([query_text], normalize_embeddings=True)
    qv = qv.astype(np.float32)[0]

    h5py = require_h5py()
    with h5py.File(h5_path, "r") as h5:
        embs = h5["chunks/embedding"]
        texts = h5["chunks/text"]
        work_ids = h5["chunks/work_id"]
        works_idx = load_works_index(h5)

        if work_filter:
            bounds = find_work_rows(works_idx, work_filter)
            if not bounds:
                raise ValueError(f'No work with title containing "{work_filter}"')
            lo, hi = bounds
            rows = np.arange(lo, hi, dtype=np.int64)
        else:
            rows = np.arange(embs.shape[0], dtype=np.int64)

        block = 131072
        scores = np.empty(rows.shape[0], dtype=np.float32)
        for i in range(0, rows.shape[0], block):
            sl = rows[i : i + block]
            matrix = np.asarray(embs[sl, :], dtype=np.float32)
            scores[i : i + len(sl)] = matrix @ qv

        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        results = []
        for rank, i_local in enumerate(idx, start=1):
            row = int(rows[i_local])
            score = float(scores[i_local])
            work_id = int(work_ids[row])
            work_title = next(
                (rec["title"] for rec in works_idx if rec["work_id"] == work_id),
                f"work_id={work_id}",
            )
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "row": row,
                    "work_id": work_id,
                    "work_title": work_title,
                    "text": str(texts[row]),
                }
            )
        return results


def cli():
    p = argparse.ArgumentParser(description="SEEK records -> HDF5 vector DB (+ query)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build the HDF5 (skips if exists unless --rebuild)")
    pb.add_argument("--input", default="sample.txt", help="Input text file of SEEK records")
    pb.add_argument("--out", required=True, help="Output HDF5 path")
    pb.add_argument("--rebuild", action="store_true", help="Force rebuild if file exists")

    pl = sub.add_parser("list-works", help="List indexed records from an existing HDF5")
    pl.add_argument("--h5", required=True, help="Path to HDF5")

    pq = sub.add_parser("query", help="Query an existing HDF5 (cosine similarity)")
    pq.add_argument("--h5", required=True)
    pq.add_argument("--text", required=True, help="Query text")
    pq.add_argument(
        "--work",
        default=None,
        help='Restrict to a record (QID or label substring), e.g. "Q13955" or "Arabic"',
    )
    pq.add_argument("--top-k", type=int, default=5)
    pq.add_argument(
        "--model",
        default=EMBEDDER_NAME,
        help="Embedding model for the query (defaults to the one used at build time)",
    )

    args = p.parse_args()

    if args.cmd == "build":
        build_if_missing(args.out, input_path=args.input, rebuild=args.rebuild)

    elif args.cmd == "list-works":
        if not os.path.exists(args.h5):
            print(f"File not found: {args.h5}", file=sys.stderr)
            sys.exit(1)
        h5py = require_h5py()
        with h5py.File(args.h5, "r") as h5:
            rows = load_works_index(h5)
        print(f"{'ID':>4} | {'Start':>7} | {'End':>7} | Title")
        print("-" * 100)
        for rec in rows:
            print(
                f"{rec['work_id']:>4} | {rec['start_row']:>7} | "
                f"{rec['end_row']:>7} | {rec['title']}"
            )

    elif args.cmd == "query":
        results = query_hdf5(
            args.h5,
            args.text,
            top_k=args.top_k,
            work_filter=args.work,
            model_name=args.model,
        )
        for r in results:
            print(
                f"[{r['rank']:>2}] score={r['score']:.4f} row={r['row']} "
                f"work_id={r['work_id']} title={r['work_title']}"
            )
            snippet = r["text"].replace("\n", " ")
            print("     ", (snippet[:180] + "...") if len(snippet) > 180 else snippet)


if __name__ == "__main__":
    cli()
