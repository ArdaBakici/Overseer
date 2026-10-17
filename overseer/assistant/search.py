"""What the assistant adds to the search (core/search.py): the vault's notes, and how a hit of theirs reads."""

from __future__ import annotations

from ..core.notify import code, esc
from ..core.search import _when

MAX_NOTES = 5000
MAX_BYTES = 30_000  # of each note (its beginning)


def search_rows(domain) -> list[tuple]:
    vault = domain.vault
    rows = []
    notes = vault.notes()
    for p in sorted(notes, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:MAX_NOTES]:
        try:
            if p.is_symlink():
                continue
            with open(p, "rb") as fh:
                body = fh.read(MAX_BYTES).decode(errors="replace")
            mtime = p.stat().st_mtime
        except OSError:
            continue
        rel = vault.rel(p)
        rows.append((body, rel[:-3] if rel.endswith(".md") else rel, "note", rel, "", "", mtime, ""))
    return rows


def search_line(h, where: str, snippet: str, html: bool) -> str | None:
    if h.kind != "note":
        return None
    e = esc if html else (lambda x: x)
    c = code if html else (lambda x: x)
    return f"📝 {c(h.ref)} · {e(_when(h.ts)[:10])}: {snippet}"
