"""
Rebuild whole Wikipedia articles from the pre-chunked psgs_w100 dump.

One-off preprocessing so that chunking can be a benchmarked component instead of
a decision inherited from DPR. Streams the dump, groups passages by article, and
writes one JSON record per article: {wikipedia_id, wikipedia_title, text,
n_passages}. Constant memory — passages of an article are contiguous, so a group
is flushed as soon as the id changes.

    python -m rag_pipeline.build_article_corpus --limit 50000 --verify
    python -m rag_pipeline.build_article_corpus

Three things about this dump are not obvious, and all three were measured on the
real file. Full reasoning is in the thesis caveats; the short version:

  * The lead passage of each article repeats the title on its own line, and the
    title is HTML-escaped there ("Jammu &amp; Kashmir") but not in the
    wikipedia_title column — so the match has to unescape before comparing.
  * DPR did NOT split on whitespace. ~15% of non-lead passages start with
    punctuation, so joining with a space yields "TAI ," — text that was never in
    Wikipedia. Those seams join with no separator.
  * A passage boundary landing on a paragraph break must join with a newline or
    two paragraphs merge. `end_paragraph` finds these exactly: a passage carries
    one newline per paragraph transition inside it, so FEWER newlines than the
    end_paragraph delta means the transition was consumed at the seam.

`--verify` compares the output to its source passages character by character and
counts spaces wrongly introduced before punctuation. It has caught two real bugs
in this file; run it after changing anything here.
"""

import argparse
import csv
import html
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 ** 7)

# A passage opening with one of these continues the previous passage's last
# token, so the seam takes no separator. Quotes are excluded: " and ' open as
# easily as they close, and guessing wrong would delete a real space.
NO_SPACE_BEFORE = ",.;:!?)]}%…"

# A space before closing punctuation — what a wrong join introduces. Newlines are
# excluded on purpose: a paragraph seam whose next passage opens with punctuation
# is a source artefact (template stripping leaves "\n, a total of 552 people"),
# and the newline there is the correct join.
SPACED_PUNCT = re.compile(r"[ \t][,.;:!?)\]}%…]")


def strip_title(text: str, title: str) -> tuple[str, bool]:
    """
    Drop the repeated title line from a lead passage.

    Compares unescaped but slices the RAW text, so the surviving bytes stay
    exactly as indexed. Returns ("", True) for a stub that was only its title.
    """
    if not title:
        return text, False
    first_line, _, rest = text.partition("\n")
    if first_line == title or html.unescape(first_line).strip() == title:
        return rest, True
    return text, False


def reconstruct(rows: list[dict], stats: Counter) -> str:
    """Join one article's passages back into a single body."""
    title = (rows[0].get("wikipedia_title") or "").strip()
    text, matched = strip_title(rows[0]["text"], title)
    stats["title_stripped" if matched else "title_absent"] += 1

    parts = [text]
    prev_para = _int(rows[0].get("end_paragraph"))

    for row in rows[1:]:
        body = row["text"]
        para = _int(row.get("end_paragraph"))

        if para is None or prev_para is None:
            # Nothing to reason from. A space loses a paragraph break at worst,
            # where a newline would invent one.
            sep, kind = " ", "seam_space"
        else:
            delta, newlines = para - prev_para, body.count("\n")
            if newlines < delta:
                sep, kind = "\n", "seam_newline"
            elif newlines == delta:
                sep, kind = " ", "seam_space"
            else:
                sep, kind = " ", "seam_anomalous"

        # Applied after the paragraph rule, never before: a real paragraph break
        # still wins over a passage that merely opens with punctuation.
        if sep == " " and body[:1] in NO_SPACE_BEFORE:
            sep, kind = "", "seam_tight"

        stats[kind] += 1
        parts += [sep, body]
        if para is not None:
            prev_para = para

    return "".join(parts)


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_articles(path: str, limit: int | None, stats: Counter):
    """Stream the dump, yielding one article's rows at a time."""
    current, rows = None, []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if limit is not None and stats["passages"] >= limit:
                break
            stats["passages"] += 1
            if not (row.get("text") or "").strip() or not row.get("wikipedia_id"):
                stats["rows_skipped"] += 1
                continue
            if row["wikipedia_id"] != current:
                if rows:
                    yield rows
                current, rows = row["wikipedia_id"], []
            rows.append(row)
    if rows:
        yield rows


def verify(rows: list[dict], article: str) -> tuple[str | None, int]:
    """
    Check one article against its source passages.

    Returns (problem_or_None, spurious_spaces).

    Compares NON-WHITESPACE CHARACTERS, not words. A word-based comparison was
    blind to the bug that shipped here: joining a mid-token seam with a space
    turned "TAI," into "TAI ,", which splits to the same two words and compared
    equal while the text was wrong. Characters catch any alteration and stay
    insensitive to the separator, which is the one thing this script decides —
    so the separator is checked separately, by counting spaces introduced before
    punctuation that the source did not have.
    """
    sources = []
    for i, row in enumerate(rows):
        text = row["text"]
        if i == 0:
            text, _ = strip_title(text, (row.get("wikipedia_title") or "").strip())
        sources.append(text)

    original = "".join("".join(s.split()) for s in sources)
    rebuilt = "".join(article.split())

    problem = None
    if rebuilt != original:
        if len(rebuilt) != len(original):
            problem = f"character count {len(rebuilt)} != {len(original)}"
        else:
            i = next(i for i, (a, b) in enumerate(zip(rebuilt, original)) if a != b)
            problem = f"character {i} differs: {rebuilt[i]!r} != {original[i]!r}"

    spurious = (len(SPACED_PUNCT.findall(article))
                - sum(len(SPACED_PUNCT.findall(s)) for s in sources))
    return problem, spurious


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag_pipeline.build_article_corpus",
        description="Rebuild whole articles from the psgs_w100 passage dump.",
    )
    parser.add_argument("--input", default="data/wikiDump/psgs_w100.tsv")
    parser.add_argument("--output", default="data/wikiDump/articles.jsonl")
    parser.add_argument("--limit", type=int, help="stop after N passages (smoke test)")
    parser.add_argument("--verify", type=int, nargs="?", const=2000, default=0,
                        metavar="N", help="check the first N articles (default 2000)")
    args = parser.parse_args(argv)

    # The summary uses box-drawing characters that a default Windows console
    # (cp1252) cannot encode. Fine on the cluster, fatal locally.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if not Path(args.input).exists():
        print(f"[articles] input not found: {args.input}")
        return 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    failures: list[str] = []
    spurious = checked = 0
    start = time.time()
    print(f"[articles] {args.input} -> {args.output}"
          + (f"  (limit {args.limit:,} passages)" if args.limit else ""))

    with open(args.output, "w", encoding="utf-8") as out:
        for rows in iter_articles(args.input, args.limit, stats):
            text = reconstruct(rows, stats)
            if not text.strip():
                # A stub whose only content was its title has nothing to chunk.
                stats["stubs_dropped"] += 1
                continue

            if checked < args.verify:
                problem, extra_spaces = verify(rows, text)
                checked += 1
                spurious += extra_spaces
                if problem:
                    failures.append(f"article {rows[0]['wikipedia_id']}: {problem}")

            out.write(json.dumps({
                "wikipedia_id": rows[0]["wikipedia_id"],
                "wikipedia_title": rows[0].get("wikipedia_title") or "",
                "text": text,
                "n_passages": len(rows),
            }, ensure_ascii=False) + "\n")
            stats["articles"] += 1

            if stats["articles"] % 250_000 == 0:
                print(f"  {stats['passages']:,} passages -> {stats['articles']:,} articles...")

    return _report(stats, failures, spurious, checked, time.time() - start)


def _report(stats, failures, spurious, checked, elapsed) -> int:
    seams = sum(stats[k] for k in
                ("seam_space", "seam_newline", "seam_tight", "seam_anomalous"))
    pct = lambda k: f"{stats[k] / seams * 100:4.1f}%" if seams else "   n/a"
    sep = "─" * 54

    print("\n".join([
        sep,
        f"  Passages / articles : {stats['passages']:,} / {stats['articles']:,}"
        f"   ({stats['rows_skipped']:,} rows skipped, {stats['stubs_dropped']:,} stubs)",
        f"  Title line          : {stats['title_stripped']:,} stripped, "
        f"{stats['title_absent']:,} absent  <- absent should be ~0",
        f"  Seams               : {stats['seam_space']:,} space  "
        f"{stats['seam_newline']:,} newline ({pct('seam_newline')})  "
        f"{stats['seam_tight']:,} tight ({pct('seam_tight')})  "
        f"{stats['seam_anomalous']:,} anomalous",
        f"  Elapsed             : {elapsed:.1f} s",
    ]))

    if checked:
        print(f"  Verified            : {checked - len(failures)}/{checked} articles exact, "
              f"{spurious:,} spurious spaces (must be 0)")
        for failure in failures[:10]:
            print(f"    FAIL {failure}")
    print(sep)

    return 2 if (failures or spurious) else 0


if __name__ == "__main__":
    raise SystemExit(main())
