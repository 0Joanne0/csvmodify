#!/usr/bin/env python3
"""
Normalize multi-valued skill columns in a semicolon-separated CSV.

Designed for the dataset in `data_jobs.csv` where columns are:
- Soft_Skills
- Hard_Skills
- Benefits

What it does
------------
- Splits each cell into items (comma/pipe/slash separated).
- Normalizes text (case/accents/punctuation/underscore vs space).
- Removes duplicates, including order-permutation duplicates (e.g. "python programming" vs "programming python").
- Collapses variants to a canonical base skill when possible (e.g. any phrase containing token "python" -> "python"
  if "python" exists as a learned base skill in the column).

Output
------
By default it writes new columns:
- Soft_Skills_Normalized
- Hard_Skills_Normalized
- Benefits_Normalized

Values are joined with ", " and items are emitted in snake_case (spaces -> underscores).
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


MISSING_STRINGS = {
    "",
    "non spécifié",
    "non specifie",
    "non determiné",
    "non determine",
    "nan",
    "none",
    "null",
}


_RE_MULTI_SPLIT = re.compile(r"\s*(?:,|\||/|\n|\t)+\s*")
_RE_KEEP_CHARS = re.compile(r"[^0-9a-zA-Z+#.\s]+")
_RE_WS = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    # NFKD splits accent marks into combining chars, which we drop.
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def normalize_text(s: str) -> str:
    """
    Normalize a skill/benefit item into a comparable form.
    - lower
    - remove accents
    - treat '_' as space
    - keep alphanum plus '+', '#', '.'
    - collapse whitespace
    """
    s = (s or "").strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = s.replace("_", " ")
    # Replace most punctuation with spaces, but keep + # . (useful for c++, c#, .net)
    s = _RE_KEEP_CHARS.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    # Normalize common tech tokens
    if s in {".net", "dot net"}:
        return "dotnet"
    return s


def to_snake(s: str) -> str:
    s = normalize_text(s)
    if not s:
        return ""
    return s.replace(" ", "_")


def is_missing_cell(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = normalize_text(value)
    return v in MISSING_STRINGS


def parse_multi_value_cell(value: Optional[str]) -> list[str]:
    """
    Parse a multi-valued cell into a list of raw items.

    Supported formats:
    - Comma-separated strings: "python, sql, power_bi"
    - Pipe-separated or slash-separated
    - Python-list-like strings: "['python', 'sql']"
    """
    if value is None:
        return []
    raw = value.strip()
    if not raw:
        return []
    if is_missing_cell(raw):
        return []

    # Try Python-list literal
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                out: list[str] = []
                for x in parsed:
                    if x is None:
                        continue
                    sx = str(x).strip()
                    if sx and not is_missing_cell(sx):
                        out.append(sx)
                return out
        except Exception:
            # Fall back to splitting
            pass

    parts = [p.strip() for p in _RE_MULTI_SPLIT.split(raw) if p and p.strip()]
    return [p for p in parts if not is_missing_cell(p)]


def token_signature(norm: str) -> tuple[str, ...]:
    """
    Signature used for permutation-insensitive dedup.
    """
    if not norm:
        return tuple()
    toks = norm.split()
    return tuple(sorted(toks))


@dataclass(frozen=True)
class BasePhrase:
    norm: str
    tokens: frozenset[str]
    count: int


def learn_base_phrases(
    items: Iterable[str],
    *,
    min_count: int = 8,
    max_tokens: int = 3,
) -> list[BasePhrase]:
    """
    Learn base phrases from observed items.

    We keep frequent, short normalized phrases. These become canonical targets when their tokens
    appear inside a longer phrase (subset match).
    """
    counts: Counter[str] = Counter()
    for it in items:
        n = normalize_text(it)
        if not n:
            continue
        counts[n] += 1

    bases: list[BasePhrase] = []
    for n, c in counts.items():
        toks = n.split()
        if c < min_count:
            continue
        if len(toks) == 0 or len(toks) > max_tokens:
            continue
        bases.append(BasePhrase(norm=n, tokens=frozenset(toks), count=c))

    # Prefer more specific (more tokens), then more frequent.
    bases.sort(key=lambda b: (len(b.tokens), b.count, b.norm), reverse=True)
    return bases


def build_inverted_index(bases: list[BasePhrase]) -> dict[str, list[BasePhrase]]:
    idx: dict[str, list[BasePhrase]] = defaultdict(list)
    for b in bases:
        for t in b.tokens:
            idx[t].append(b)
    return idx


def choose_best_base(norm_phrase: str, idx: dict[str, list[BasePhrase]]) -> Optional[BasePhrase]:
    toks = norm_phrase.split()
    if not toks:
        return None
    tokset = set(toks)

    candidates: dict[str, BasePhrase] = {}
    for t in tokset:
        for b in idx.get(t, []):
            candidates[b.norm] = b

    best: Optional[BasePhrase] = None
    for b in candidates.values():
        if not b.tokens.issubset(tokset):
            continue
        if best is None:
            best = b
            continue
        # Prefer base with more tokens; if equal, prefer higher count.
        if len(b.tokens) > len(best.tokens):
            best = b
        elif len(b.tokens) == len(best.tokens) and b.count > best.count:
            best = b
    return best


def normalize_items(
    raw_items: list[str],
    *,
    base_index: Optional[dict[str, list[BasePhrase]]] = None,
    emit_snake_case: bool = True,
) -> list[str]:
    """
    Normalize + deduplicate a list of items.
    """
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()

    for raw in raw_items:
        norm = normalize_text(raw)
        if not norm:
            continue

        if base_index:
            base = choose_best_base(norm, base_index)
            if base is not None:
                norm = base.norm

        sig = token_signature(norm)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        out.append(to_snake(norm) if emit_snake_case else norm)

    return out


def iter_column_items(csv_path: Path, column: str, *, delimiter: str) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        # Handle BOM on first field name
        if reader.fieldnames:
            reader.fieldnames = [fn.lstrip("\ufeff") for fn in reader.fieldnames]
        for row in reader:
            cell = row.get(column)
            for it in parse_multi_value_cell(cell):
                yield it


def normalize_csv(
    *,
    input_path: Path,
    output_path: Path,
    delimiter: str = ";",
    columns: tuple[str, str, str] = ("Hard_Skills", "Soft_Skills", "Benefits"),
    min_base_count: int = 8,
    max_base_tokens: int = 3,
    add_new_columns: bool = True,
    inplace: bool = False,
) -> None:
    hard_col, soft_col, ben_col = columns

    # Pass 1: learn bases per column (so "python programming" can collapse to "python" if "python" is frequent).
    hard_bases = learn_base_phrases(
        iter_column_items(input_path, hard_col, delimiter=delimiter),
        min_count=min_base_count,
        max_tokens=max_base_tokens,
    )
    soft_bases = learn_base_phrases(
        iter_column_items(input_path, soft_col, delimiter=delimiter),
        min_count=min_base_count,
        max_tokens=max_base_tokens,
    )
    ben_bases = learn_base_phrases(
        iter_column_items(input_path, ben_col, delimiter=delimiter),
        min_count=min_base_count,
        max_tokens=max_base_tokens,
    )

    hard_idx = build_inverted_index(hard_bases)
    soft_idx = build_inverted_index(soft_bases)
    ben_idx = build_inverted_index(ben_bases)

    # Pass 2: normalize and write
    with input_path.open("r", encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter=delimiter)
        if reader.fieldnames:
            reader.fieldnames = [fn.lstrip("\ufeff") for fn in reader.fieldnames]

        fieldnames = list(reader.fieldnames or [])
        hard_out = f"{hard_col}_Normalized"
        soft_out = f"{soft_col}_Normalized"
        ben_out = f"{ben_col}_Normalized"

        if inplace:
            # Replace originals
            pass
        elif add_new_columns:
            for c in (soft_out, hard_out, ben_out):
                if c not in fieldnames:
                    fieldnames.append(c)

        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()

        for row in reader:
            # Normalize per column
            hard_items = normalize_items(parse_multi_value_cell(row.get(hard_col)), base_index=hard_idx)
            soft_items = normalize_items(parse_multi_value_cell(row.get(soft_col)), base_index=soft_idx)
            ben_items = normalize_items(parse_multi_value_cell(row.get(ben_col)), base_index=ben_idx)

            hard_joined = ", ".join(hard_items) if hard_items else ""
            soft_joined = ", ".join(soft_items) if soft_items else ""
            ben_joined = ", ".join(ben_items) if ben_items else ""

            if inplace:
                row[hard_col] = hard_joined
                row[soft_col] = soft_joined
                row[ben_col] = ben_joined
            elif add_new_columns:
                row[hard_out] = hard_joined
                row[soft_out] = soft_joined
                row[ben_out] = ben_joined

            writer.writerow(row)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Normalize Hard_Skills / Soft_Skills / Benefits columns in a CSV.")
    p.add_argument("--input", required=True, type=Path, help="Input CSV path (semicolon-separated).")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    p.add_argument("--delimiter", default=";", help="CSV delimiter (default: ';').")
    p.add_argument("--hard-col", default="Hard_Skills", help="Hard skills column name.")
    p.add_argument("--soft-col", default="Soft_Skills", help="Soft skills column name.")
    p.add_argument("--benefits-col", default="Benefits", help="Benefits column name.")
    p.add_argument("--min-base-count", type=int, default=8, help="Min frequency to treat an item as canonical base.")
    p.add_argument("--max-base-tokens", type=int, default=3, help="Max tokens for canonical base phrases.")
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Replace original columns instead of creating *_Normalized columns.",
    )
    args = p.parse_args(argv)

    normalize_csv(
        input_path=args.input,
        output_path=args.output,
        delimiter=args.delimiter,
        columns=(args.hard_col, args.soft_col, args.benefits_col),
        min_base_count=args.min_base_count,
        max_base_tokens=args.max_base_tokens,
        inplace=args.inplace,
        add_new_columns=not args.inplace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

