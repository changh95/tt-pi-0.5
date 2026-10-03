"""Integrity of an STE draft against its published source: front matter (if any) and code blocks identical, the same
set of number values (none lost, none new), the same set of `code` spans, URLs and hex ids (7+ hex chars).
usage: integrity_check.py PUBLISHED.md DRAFT.md"""
import re, sys
a, b = [open(x).read() for x in sys.argv[1:3]]
ok = True
def rep(name, cond, detail=""):
    global ok
    ok &= bool(cond); print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")
fm = lambda t: t.split("---\n", 2)[1] if t.startswith("---\n") else None
rep("front matter identical", fm(a) == fm(b))
code = lambda t: re.findall(r"```.*?```", t, flags=re.S)
rep("code blocks identical", code(a) == code(b), f"({len(code(a))} vs {len(code(b))})")
strip = lambda t: re.sub(r"```.*?```", "", t, flags=re.S)
num = lambda t: set(re.findall(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?", strip(t)))
na, nb = num(a), num(b)
rep("no number lost", not (na - nb), sorted(na - nb)[:40])
rep("no new number", not (nb - na), sorted(nb - na)[:40])
spans = lambda t: set(re.findall(r"`([^`\n]+)`", strip(t)))
lost = sorted(spans(a) - spans(b))
rep("no `code` span lost", not lost, lost[:30])
urls = lambda t: set(re.findall(r"https?://[^\s)>\"']+", t))
rep("URLs identical", urls(a) == urls(b), sorted(urls(a) ^ urls(b))[:10])
hx = lambda t: set(re.findall(r"\b[0-9a-f]{7,64}\b", t))
rep("hex ids identical", hx(a) == hx(b), sorted(hx(a) ^ hx(b))[:10])
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
