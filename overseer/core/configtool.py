"""`overseer config`: bring an old config file up to date with the current example.

It reports what you changed from the defaults, the settings that are new since your file (with the example's
explanation), and the ones no longer used; and writes <file>.new: the current example (its comments and order)
with your values in it. Settings you never set keep their default (not the example's suggestion), so the new
file means exactly what your old one did - which is checked before it's written. Your file is never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import Config, ConfigError, _build, read_yaml

KEY_RE = re.compile(r"^(\s*)([A-Za-z_][\w-]*):(.*)$")
Path_ = tuple[str, ...]


def defaults(domain: str) -> dict[str, Any]:
    cfg = Config(domain=domain)
    out: dict[str, Any] = {}
    for f in fields(Config):
        if f.name == "sections":
            continue
        v = getattr(cfg, f.name)
        out[f.name] = asdict(v) if is_dataclass(v) else v
    for name, section in cfg.sections.items():
        out[name] = asdict(section)
    out["local"].pop("base_dir", None)
    return out


def leaves(data: dict, guide: dict, prefix: Path_ = ()) -> dict[Path_, Any]:
    """The settings in `data`, by path - as deep as the defaults (`guide`) have named fields."""
    out = {}
    for k, v in (data or {}).items():
        path = (*prefix, str(k))
        g = guide.get(k) if isinstance(guide, dict) else None
        if isinstance(v, dict) and isinstance(g, dict) and g and _is_section(g):
            out.update(leaves(v, g, path))
        else:
            out[path] = v
    return out


def _is_section(d: dict) -> bool:
    """A section of named settings (its defaults have fixed keys), not a free mapping like projects: {}."""
    return bool(d) and all(isinstance(k, str) for k in d)


def _known(path: Path_, guide: dict) -> bool:
    node: Any = guide
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


def fmt(value: Any) -> str:
    """A value as the example files write them: strings plain unless they need quotes (then "double"), lists
    inline with their strings quoted."""
    if isinstance(value, str):
        s = _dump(value)
        return json.dumps(value, ensure_ascii=False) if s.startswith(("'", '"')) or not s else s
    if isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value):
        return "[" + ", ".join(json.dumps(x, ensure_ascii=False) if isinstance(x, str) else _dump(x)
                               for x in value) + "]"
    if isinstance(value, dict) and value:
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) + "}"
    return _dump(value)


def _dump(value: Any) -> str:
    s = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, sort_keys=False, width=10**6).strip()
    return s[:-4].strip() if s.endswith("\n...") else s


def _split_value(rest: str) -> tuple[str, str]:
    """' value   # comment' -> (value, '   # comment'), quotes respected."""
    quote = None
    for i, c in enumerate(rest):
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "#" and i > 0 and rest[i - 1].isspace():
            j = i
            while j > 0 and rest[j - 1].isspace():
                j -= 1
            return rest[:j].strip(), rest[j:]
    return rest.strip(), ""


def _block(key: str, indent: str, value: Any, comment: str = "") -> list[str]:
    """A mapping (or a list of them) as a YAML block under `key`."""
    body = yaml.safe_dump(value, default_flow_style=False, allow_unicode=True, sort_keys=False, width=10**6)
    return [f"{indent}{key}:{comment}"] + [f"{indent}  {line}" for line in body.rstrip("\n").split("\n")]


def _value_lines(key: str, indent: str, value: Any, comment: str = "", column: int = 0) -> list[str]:
    flat = isinstance(value, dict) and all(not isinstance(v, (dict, list)) or isinstance(v, list) and
                                           all(not isinstance(x, (dict, list)) for x in v) for v in value.values())
    if isinstance(value, dict) and value and not (flat and len(fmt(value)) < 80) \
            or (isinstance(value, list) and any(isinstance(x, dict) for x in value)):
        return _block(key, indent, value, comment)
    line = f"{indent}{key}: {fmt(value)}"
    if comment:  # (the comment stays in its column if the value fits)
        line = line.ljust(column) if len(line) < column else line + " "
        line += comment.lstrip()
    return [line]


def merge(example_text: str, mine: dict[Path_, Any], defaults_: dict[Path_, Any]) -> str:
    """The example's text with these settings in it (and the defaults where neither sets one)."""
    out: list[str] = []
    stack: list[tuple[int, str, int]] = []  # (indent, key, index in out of its line)
    parents: dict[Path_, tuple[int, int]] = {}  # mapping path -> (indent, index of its line)
    used: set[Path_] = set()
    for line in example_text.split("\n"):
        m = KEY_RE.match(line)
        if not m or line.lstrip().startswith("#"):
            out.append(line)
            continue
        indent, key, rest = m.group(1), m.group(2), m.group(3)
        while stack and stack[-1][0] >= len(indent):
            stack.pop()
        path = (*(k for _, k, _ in stack), key)
        value, comment = _split_value(rest)
        column = len(line) - len(comment.lstrip()) if comment else 0
        try:
            shown = yaml.safe_load(value) if value else None
        except yaml.YAMLError:
            shown = object()
        if not value and path not in defaults_:  # a section (its settings follow)
            stack.append((len(indent), key, len(out)))
            parents[path] = (len(indent), len(out))
            out.append(line)
            continue
        inner = {p: v for src in (defaults_, mine) for p, v in src.items() if p[:len(path)] == path and p != path}
        if isinstance(shown, dict) and inner:  # an inline mapping (e.g. planner: {claude_code: ...}): as a whole
            want: Any = {}
            for p, v in inner.items():
                node = want
                for k in p[len(path):-1]:
                    node = node.setdefault(k, {})
                node[p[-1]] = v
            used.update(inner)
        else:
            want = mine[path] if path in mine else defaults_.get(path, shown)
        used.add(path)
        if want == shown:  # (the example's own line, as it is)
            out.append(line)
        else:
            out += _value_lines(key, indent, want, comment, column)
    # Your settings the example doesn't show (e.g. pass_env): right under their section, else at the end.
    inserts: dict[int, list[str]] = {}
    tail: list[str] = []
    for path, value in mine.items():
        if path in used:
            continue
        for depth in range(len(path) - 1, 0, -1):
            if path[:depth] in parents:
                ind, at = parents[path[:depth]]
                lines, pad = [], " " * (ind + 2)
                for k in path[depth:-1]:  # (sections the example doesn't have)
                    lines.append(f"{pad}{k}:")
                    pad += "  "
                inserts.setdefault(at, []).extend(lines + _value_lines(path[-1], pad, value))
                break
        else:
            node: Any = value
            for k in reversed(path[1:]):
                node = {k: node}
            tail += _value_lines(path[0], "", node)
    merged = []
    for i, line in enumerate(out):
        merged.append(line)
        merged += inserts.get(i, [])
    text = "\n".join(merged).rstrip("\n") + "\n"
    return text + ("\n" + "\n".join(tail) + "\n" if tail else "")


def example_path(domain: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    own = root / f"{domain}.example.yaml"
    return own if own.exists() else root / "config.example.yaml"


def update(config_path: Path) -> tuple[str, str | None]:
    """(the report, the new file's text or None if there's nothing to change)."""
    raw = read_yaml(config_path)
    domain = str(raw.get("domain") or "hpc")
    example = example_path(domain)
    guide = defaults(domain)
    default_leaves = leaves(guide, guide)
    mine_all = leaves(raw, guide)
    mine = {p: v for p, v in mine_all.items() if _known(p, guide)}
    unused = [p for p in mine_all if p not in mine]
    example_leaves = leaves(yaml.safe_load(example.read_text()) or {}, guide)

    changed = [(p, v, default_leaves.get(p)) for p, v in mine.items() if default_leaves.get(p, object()) != v]
    new = [p for p in example_leaves if p not in mine and p in default_leaves]
    lines = [f"{config_path} (domain: {domain}), compared with {example.name}:", ""]
    lines.append("Your settings (different from the defaults):" if changed else "You changed no defaults.")
    for p, v, d in changed:
        lines.append(f"  {'.'.join(p)}: {fmt(v)}" + (f"   (default: {fmt(d)})" if p in default_leaves else ""))
    if new:
        lines += ["", "Settings your file doesn't have (they keep their defaults; the example's suggestion in []):"]
        for p in new:
            suggestion = example_leaves[p]
            lines.append(f"  {'.'.join(p)}: {fmt(default_leaves[p])}"
                         + (f"   [{fmt(suggestion)}]" if suggestion != default_leaves[p] else ""))
    if unused:
        lines += ["", "In your file but not used any more (left out of the new file):"]
        lines += [f"  {'.'.join(p)}: {fmt(mine_all[p])}" for p in unused]

    text = merge(example.read_text(), mine, default_leaves)
    clean = dict(raw)
    for p in unused:  # (what it should mean: your file without the settings that no longer exist)
        node = clean
        for k in p[:-1]:
            node[k] = dict(node.get(k) or {})
            node = node[k]
        node.pop(p[-1], None)
    if _build(Config, yaml.safe_load(text), "") != _build(Config, clean, ""):
        raise ConfigError("couldn't merge it safely (the result would mean something else): compare the files "
                          f"by hand with {example.name}")
    return "\n".join(lines), text
