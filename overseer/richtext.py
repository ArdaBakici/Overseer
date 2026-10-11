"""Rich Telegram messages: Markdown -> Telegram HTML, tag-safe splitting, collapsible sections.

Telegram's HTML is a small subset (b, i, s, u, code, pre, a, blockquote, tg-spoiler) with nesting
rules: blockquotes can't nest, and code/pre/links can't sit inside other entities except bold-like
ones. The renderer only produces combinations that are allowed: inside quotes, links and
bold/italic spans, code is shown as plain text. If Telegram still rejects a message, the notifier
falls back to plain text (strip_html).
"""

from __future__ import annotations

import html
import re

from markdown_it import MarkdownIt

TG_LIMIT = 4096
FOLD_OVER = 1200  # agent answers longer than this get their tail folded, when it can be

_md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
_LINK_SCHEMES = ("http://", "https://", "tg://", "mailto:")


def esc(s: object) -> str:
    return html.escape(str(s), quote=False)


def strip_html(text: str) -> str:
    """Telegram HTML -> plain text (for the fallback when Telegram rejects the formatting)."""
    text = re.sub(r"<br\s*/?>", "\n", text)
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def fold(inner_html: str) -> str:
    """A collapsible section: Telegram shows its first lines and expands it on tap.
    It may only contain plain text and bold/italic-like formatting."""
    inner_html = inner_html.strip("\n")
    return f"<blockquote expandable>{inner_html}</blockquote>" if inner_html else ""


def fold_text(text: str, limit: int = 3000) -> str:
    text = text.strip("\n")
    if len(text) > limit:
        text = "…\n" + text[-limit:]
    return fold(esc(text))


# ============================================================================ Markdown -> HTML

def _plain_inline(children) -> str:
    out = []
    for c in children or []:
        if c.type in ("text", "code_inline", "html_inline"):
            out.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif c.type == "image":
            out.append(c.content or "[image]")
    return "".join(out)


def _inline(children, in_quote: bool) -> str:
    out: list[str] = []
    links: list[bool] = []  # per open link: did we emit <a>?
    styles = 0  # open bold/italic/strike spans
    for c in children or []:
        t = c.type
        if t == "text":
            out.append(esc(c.content))
        elif t in ("softbreak", "hardbreak"):
            out.append("\n")
        elif t in ("strong_open", "em_open", "s_open"):
            out.append({"strong_open": "<b>", "em_open": "<i>", "s_open": "<s>"}[t])
            styles += 1
        elif t in ("strong_close", "em_close", "s_close"):
            out.append({"strong_close": "</b>", "em_close": "</i>", "s_close": "</s>"}[t])
            styles -= 1
        elif t == "code_inline":
            # code can't be nested in quotes, links or (to be safe) styled spans
            plain = in_quote or styles > 0 or any(links)
            out.append(esc(c.content) if plain else f"<code>{esc(c.content)}</code>")
        elif t == "link_open":
            href = c.attrGet("href") or ""
            ok = not in_quote and not any(links) and href.startswith(_LINK_SCHEMES)
            links.append(ok)
            if ok:
                out.append(f'<a href="{html.escape(href, quote=True)}">')
        elif t == "link_close":
            if links and links.pop():
                out.append("</a>")
        elif t == "image":
            out.append(esc(c.content or "[image]"))
        elif t == "html_inline":
            out.append(esc(c.content))
    return "".join(out)


def _table(tokens, i: int) -> tuple[str, int]:
    """Render a Markdown table as aligned monospace text; returns (text, index after table_close)."""
    rows: list[list[str]] = []
    while i < len(tokens) and tokens[i].type != "table_close":
        t = tokens[i]
        if t.type == "tr_open":
            rows.append([])
        elif t.type == "inline" and rows:
            rows[-1].append(_plain_inline(t.children).strip())
        i += 1
    if not rows:
        return "", i + 1
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    cols = [max(len(r[c]) for r in rows) for c in range(width)]
    lines = [" │ ".join(cell.ljust(cols[c]) for c, cell in enumerate(r)).rstrip() for r in rows]
    lines.insert(1, "─┼─".join("─" * w for w in cols))
    return "\n".join(lines), i + 1


def md_to_html(text: str) -> str:
    """Render the agent's Markdown as Telegram HTML."""
    tokens = _md.parse(text or "")
    out: list[str] = []
    lists: list[dict] = []
    quote = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        typ = t.type
        if typ == "heading_open":
            out.append("<b>")
        elif typ == "heading_close":
            out.append("</b>\n\n")
        elif typ == "paragraph_close":
            out.append("\n" if t.hidden or lists else "\n\n")
        elif typ == "inline":
            out.append(_inline(t.children, quote > 0))
        elif typ in ("bullet_list_open", "ordered_list_open"):
            lists.append({"ordered": typ == "ordered_list_open", "n": int(t.attrGet("start") or 1)})
        elif typ in ("bullet_list_close", "ordered_list_close"):
            lists.pop()
            if not lists:
                out.append("\n")
        elif typ == "list_item_open":
            lst = lists[-1]
            marker = f"{lst['n']}." if lst["ordered"] else "•"
            lst["n"] += 1
            out.append("   " * (len(lists) - 1) + marker + " ")
        elif typ in ("fence", "code_block"):
            code = t.content.rstrip("\n")
            lang = (t.info or "").split()[0] if typ == "fence" and t.info else ""
            if quote:
                out.append(esc(code) + "\n")
            elif lang and re.fullmatch(r"[\w+#.-]{1,20}", lang):
                out.append(f'<pre><code class="language-{lang}">{esc(code)}</code></pre>\n\n')
            else:
                out.append(f"<pre>{esc(code)}</pre>\n\n")
        elif typ == "blockquote_open":
            if not quote:
                out.append("<blockquote>")
            quote += 1
        elif typ == "blockquote_close":
            quote -= 1
            if not quote:
                while out and out[-1].endswith("\n"):
                    out[-1] = out[-1].rstrip("\n")
                    if out[-1]:
                        break
                    out.pop()
                out.append("</blockquote>\n\n")
        elif typ == "hr":
            out.append("──────────\n\n")
        elif typ == "table_open":
            table, i = _table(tokens, i + 1)
            out.append(esc(table) + "\n\n" if quote else f"<pre>{esc(table)}</pre>\n\n")
            continue
        elif typ == "html_block":
            out.append(esc(t.content))
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def fold_long(rendered: str, keep: int = 600) -> str:
    """Keep a long answer's beginning visible and fold the rest - only if the rest is plain enough
    to sit in a collapsible quote (no code blocks, links or quotes)."""
    if len(rendered) <= FOLD_OVER:
        return rendered
    cut = rendered.find("\n\n", keep // 2)
    if cut == -1 or cut > keep * 2:
        return rendered
    head, tail = rendered[:cut], rendered[cut:].strip("\n")
    if re.search(r"<(pre|code|a |blockquote)", tail) or re.search(r"<(pre|code|blockquote)", head[-10:]):
        return rendered
    return f"{head}\n{fold(tail)}"


# ============================================================================ splitting

_ATOM = re.compile(r"<[^>]+>|&#?\w+;|\n|[^<&\n]{1,1000}|[<&]")


def _atoms(text: str, step: int):
    """Tags, entities, newlines and runs of text - long runs cut into pieces of at most `step`."""
    for atom in _ATOM.findall(text):
        if atom.startswith(("<", "&")) or len(atom) <= step:
            yield atom
        else:
            yield from (atom[k:k + step] for k in range(0, len(atom), step))


def split_html(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split Telegram HTML into messages of at most `limit` characters, preferably at line breaks,
    closing the open tags at the end of a chunk and reopening them at the start of the next."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    stack: list[tuple[str, str]] = []  # (name, opening tag)
    last_nl: tuple[int, list[tuple[str, str]]] | None = None  # (position after a newline, stack there)

    def closing(st):
        return "".join(f"</{name}>" for name, _ in reversed(st))

    def reopening(st):
        return "".join(tag for _, tag in st)

    for atom in _atoms(text, max(16, limit // 4)):
        if len(cur) + len(atom) + len(closing(stack)) > limit - 10 and strip_html(cur).strip():
            if last_nl and last_nl[0] > len(cur) // 2:
                pos, st = last_nl
                chunks.append(cur[:pos] + closing(st))
                cur = reopening(st) + cur[pos:]
            else:
                chunks.append(cur + closing(stack))
                cur = reopening(stack)
            last_nl = None
        cur += atom
        if atom.startswith("</"):
            name = atom[2:-1].strip().split()[0].lower() if len(atom) > 3 else ""
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == name:
                    del stack[k:]
                    break
        elif atom.startswith("<") and len(atom) > 2 and not atom.endswith("/>"):
            stack.append((atom[1:-1].split()[0].lower(), atom))
        elif atom == "\n":
            last_nl = (len(cur), list(stack))
    if strip_html(cur).strip():
        chunks.append(cur)
    return [c for c in chunks if strip_html(c).strip()]
