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
import json
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

# Seed bases help guarantee normalization like:
# - "python programming" / "programmation python" / "programming python" -> "python"
# even if the dataset is small and the base-learning frequency threshold isn't met.
DEFAULT_HARD_SKILL_SEEDS = [
    "python",
    "sql",
    "r",
    "excel",
    "tableau",
    "power bi",
    "snowflake",
    "dbt",
    "azure",
    "aws",
    "gcp",
    "spark",
    "docker",
    "kubernetes",
    "git",
]


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
class ExplicitMaps:
    exact: dict[str, str]
    signature: dict[tuple[str, ...], str]


def load_normalization_map(path: Optional[Path]) -> ExplicitMaps:
    """
    Load a JSON mapping of variant -> canonical.

    Example (any one permutation is enough):
    {
      "python programmation": "python"
    }

    Matching is done after `normalize_text()`. In addition to exact matches, we also build a
    permutation-insensitive mapping using the token signature of each key.
    """
    if path is None:
        return ExplicitMaps(exact={}, signature={})
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")

    exact: dict[str, str] = {}
    sig: dict[tuple[str, ...], str] = {}
    for k, v in data.items():
        nk = normalize_text(str(k))
        nv = normalize_text(str(v))
        if not nk or not nv:
            continue
        exact[nk] = nv
        sig[token_signature(nk)] = nv
    return ExplicitMaps(exact=exact, signature=sig)


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


def choose_best_base(
    norm_phrase: str,
    idx: dict[str, list[BasePhrase]],
    *,
    priority_norms: Optional[set[str]] = None,
) -> Optional[BasePhrase]:
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

        # Prefer priority bases (e.g. "python") over non-priority ones.
        b_prio = 1 if (priority_norms and b.norm in priority_norms) else 0
        best_prio = 1 if (priority_norms and best.norm in priority_norms) else 0
        if b_prio != best_prio:
            if b_prio > best_prio:
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
    priority_norms: Optional[set[str]] = None,
    explicit_maps: Optional[ExplicitMaps] = None,
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

        if explicit_maps:
            # Exact variant mapping
            norm = explicit_maps.exact.get(norm, norm)
            # Permutation-insensitive mapping (covers "programmation python" vs "python programmation")
            norm = explicit_maps.signature.get(token_signature(norm), norm)

        if base_index:
            base = choose_best_base(norm, base_index, priority_norms=priority_norms)
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
    hard_skill_seeds: Optional[list[str]] = None,
    hard_map_path: Optional[Path] = None,
    soft_map_path: Optional[Path] = None,
    benefits_map_path: Optional[Path] = None,
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
    # Add seeds (high priority) so we can normalize even on small datasets.
    seed_norms = [normalize_text(s) for s in (hard_skill_seeds or DEFAULT_HARD_SKILL_SEEDS)]
    seed_norms = [s for s in seed_norms if s]
    seed_set = set(seed_norms)
    existing_hard = {b.norm for b in hard_bases}
    for s in seed_norms:
        if s in existing_hard:
            continue
        hard_bases.append(BasePhrase(norm=s, tokens=frozenset(s.split()), count=10**9))
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

    hard_maps = load_normalization_map(hard_map_path)
    soft_maps = load_normalization_map(soft_map_path)
    ben_maps = load_normalization_map(benefits_map_path)

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
            hard_items = normalize_items(
                parse_multi_value_cell(row.get(hard_col)),
                base_index=hard_idx,
                priority_norms=seed_set,
                explicit_maps=hard_maps,
            )
            soft_items = normalize_items(
                parse_multi_value_cell(row.get(soft_col)),
                base_index=soft_idx,
                explicit_maps=soft_maps,
            )
            ben_items = normalize_items(
                parse_multi_value_cell(row.get(ben_col)),
                base_index=ben_idx,
                explicit_maps=ben_maps,
            )

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
        "--hard-seeds",
        default=",".join(DEFAULT_HARD_SKILL_SEEDS),
        help="Comma-separated seed hard-skill bases to always collapse to.",
    )
    p.add_argument("--hard-map", type=Path, default=None, help="Optional JSON mapping file for hard skills.")
    p.add_argument("--soft-map", type=Path, default=None, help="Optional JSON mapping file for soft skills.")
    p.add_argument("--benefits-map", type=Path, default=None, help="Optional JSON mapping file for benefits.")
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
        hard_skill_seeds=[s.strip() for s in str(args.hard_seeds).split(",") if s.strip()],
        hard_map_path=args.hard_map,
        soft_map_path=args.soft_map,
        benefits_map_path=args.benefits_map,
        inplace=args.inplace,
        add_new_columns=not args.inplace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

