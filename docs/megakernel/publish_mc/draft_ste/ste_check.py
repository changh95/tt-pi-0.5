"""Word-count check of the STE draft README: every prose sentence outside the front matter, code blocks, tables and
HTML, with its word count and type (instruction: limit 20; descriptive: limit 25). Paragraphs with more than 6
sentences are flagged too. Words = whitespace tokens after markdown links are reduced to their text; a `code` span is
one token as written. Prose table cells (4+ alphabetic words) are checked too (limit 25 per sentence); British
spellings are flagged (the drafts use US English). Files without YAML front matter are read whole.
usage: ste_check.py FILE.md [--all]"""
import re, sys
IMPERATIVE = {"accept", "set", "send", "use", "start", "stop", "select", "denormalise", "do", "make", "read", "open",
              "install", "run", "see", "put", "give", "keep", "close", "remove", "add", "check", "get", "type"}
raw = open(sys.argv[1]).read()
txt = raw.split("---\n", 2)[2] if raw.startswith("---\n") else raw
lines, in_code, cells = [], False, []
for l in txt.split("\n"):
    if l.strip().startswith("```"):
        in_code = not in_code; lines.append(""); continue
    if not in_code and l.lstrip().startswith("|") and not re.match(r"^\s*\|[\s:|-]+\|\s*$", l):
        cells.extend(c.strip() for c in l.strip().strip("|").split("|"))
    if in_code or l.lstrip().startswith("|") or l.lstrip().startswith("<"):
        lines.append(""); continue
    lines.append(l)
paras, cur = [], []
prose, cur_prose, deep, n_items = [], [], [], 0
for l in lines:
    s = l.strip()
    if not s or s.startswith("#"):
        if cur: paras.append(cur); cur = []
        if cur_prose: prose.append(" ".join(cur_prose)); cur_prose = []
        continue
    is_item = re.match(r"^([-*]|\d+\.)\s", s)
    if is_item:
        n_items += 1
        indent = len(l) - len(l.lstrip())
        if indent >= 4: deep.append(s[:80])  # third level or deeper
        if cur_prose: prose.append(" ".join(cur_prose)); cur_prose = []
        if cur: paras.append(cur); cur = []
    elif not cur:
        cur_prose.append(s)
    cur.append(re.sub(r"^([-*]|\d+\.)\s+", "", s))
if cur: paras.append(cur)
if cur_prose: prose.append(" ".join(cur_prose))

def clean(s):
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    return s.replace("**", "").replace("*", "")

ABBR = r"(?<!e\.g)(?<!i\.e)(?<!vs)"
SPLIT = r"(?<=[.!?])\s+(?=[A-Z`\[(])"
rows, over, long_paras = [], [], []
for p in paras:
    text = clean(" ".join(p))
    sents = [x.strip() for x in re.split(ABBR + r"(?<=[.!?])\s+(?=[A-Z`\[(†0-9])", text) if x.strip()]
    if len(sents) > 6: long_paras.append((len(sents), text[:90]))
    for s in sents:
        if s.endswith(":") and len(s.split()) <= 3:  # a label
            continue
        w = len(s.split())
        core = re.sub(r"^(Caution|Warning|Note):\s*", "", s)
        core = re.sub(r"^(Then|Also)\s+", "", core)
        core = re.sub(r"^(If|When|For|With|To|At|Before|After)\b[^,]*,\s*", "", core)
        tok = (core.split() or [""])[0]
        first = "" if tok.startswith("`") else re.sub(r"[^A-Za-z]", "", tok).lower()
        kind = "instruction" if first in IMPERATIVE else "descriptive"
        lim = 20 if kind == "instruction" else 25
        rows.append((w, kind, lim, s))
        if w > lim: over.append((w, kind, lim, s))
cell_rows, cell_over = [], []
for c in cells:
    t = clean(re.sub(r"`[^`]*`", "X", c))
    if len(re.findall(r"[A-Za-z]{2,}", t)) < 4:
        continue
    for s_ in [x.strip() for x in re.split(SPLIT, t) if x.strip()]:
        w = len(s_.split()); cell_rows.append((w, s_))
        if w > 25: cell_over.append((w, s_))
BRIT = re.compile(r"\b\w*(normalis|tokenis|recognis|organis|initialis|optimis|serialis|discretis|summaris|minimis|maximis|parallelis|behaviour|colour|centre|licence)\w*\b", re.I)
prose_only = re.sub(r"`[^`]*`", "", "\n".join(lines))
brit = sorted(set(m.group(0) for m in BRIT.finditer(prose_only + "\n" + "\n".join(re.sub(r"`[^`]*`", "", c) for c in cells))))
n_i = sum(r[1] == "instruction" for r in rows)
print(f"sentences {len(rows)} (instructions {n_i}, descriptive {len(rows) - n_i}); over limit {len(over)}; "
      f"paragraphs > 6 sentences {len(long_paras)}; max length {max(r[0] for r in rows) if rows else 0}; mean {(sum(r[0] for r in rows)/len(rows)) if rows else 0:.1f}")
for w, k, lim, s in over:
    print(f"  OVER {w}/{lim} {k}: {s}")
print(f"table prose cells: sentences {len(cell_rows)}, over 25 {len(cell_over)}" + (f", max {max(w for w, _ in cell_rows)}" if cell_rows else ""))
for w, s_ in cell_over:
    print(f"  CELL OVER {w}/25: {s_}")
print(f"British spellings: {brit}")
print(f"list items {n_items}; list items nested deeper than 2 levels {len(deep)}; prose paragraphs (not lists) {len(prose)}")
for t in prose:
    ns = len(re.split(SPLIT, clean(t)))
    print(f"  PROSE ({ns} sentence(s)): {clean(t)[:160]}")
for d in deep:
    print(f"  DEEP: {d}")
for n, t in long_paras:
    print(f"  LONG PARAGRAPH {n} sentences: {t}...")
if "--all" in sys.argv:
    for w, k, lim, s in rows:
        print(f"{w:3d}/{lim} {k[:4]} | {s}")
