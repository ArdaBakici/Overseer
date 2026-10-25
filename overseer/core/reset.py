"""`overseer reset`: wipe a bot's flags, and - if you say so - forget its conversations.

Flags: the wake-ups its agent set, paused wake-ups, what `ov react` reacts to, and the domain's own (e.g. the
assistant's follow-ups, day logs and first-run flag). Conversations: their records, transcripts and search index,
the agent CLIs' own sessions (what lets a conversation be resumed), and the agents' scratch folders. Kept either
way: notes, projects, jobs and results, settings, usage, and the CLIs' logins.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config
from .store import Store

PLATFORM_FLAGS = ("wakeups_paused:", "last_user_message:", "rewake:")
# kv entries that belong to conversations: the main chat's / projects' current one, runs cut off by a restart,
# notes for an agent's next turn, planner proposals, the feedback reviewer, questions asked with buttons (ov ask).
CONVERSATION_KEYS = ("general_session", "run:", "cutoff:", "notes:", "proposal:", "proposal_id:", "feedback_reviewer",
                     "ask:")
CLAUDE_SESSION_FILES = ("projects", "history.jsonl", "todos", "file-history", "shell-snapshots", "session-env")
CODEX_SESSION_FILES = ("sessions", "archived_sessions", "history.jsonl")


def flags(store: Store, domain) -> tuple[list[str], list]:
    keys = [k for p in (*PLATFORM_FLAGS, *domain.reset_prefixes) for k in store.keys(p)]
    return keys, store.watches()


def remove_flags(store: Store, keys: list[str], watches: list) -> None:
    for k in keys:
        store.unset(k)
    for w in watches:
        w.state = "cancelled"
        store.update_watch(w)


def conversation_files(cfg: Config) -> list[Path]:
    """The files that hold conversations (only those that exist)."""
    candidates = [cfg.local.state("transcripts"), cfg.local.state("search.db"),
                  cfg.local.state("search.db-wal"), cfg.local.state("search.db-shm"), cfg.local.tmp("agents")]
    claude = cfg.local.resolve(cfg.agent.claude_code.home)
    codex = cfg.local.resolve(cfg.agent.codex.home)
    candidates += [claude / n for n in CLAUDE_SESSION_FILES] + [codex / n for n in CODEX_SESSION_FILES]
    return [p for p in candidates if p.exists() or p.is_symlink()]


def conversations(store: Store) -> int:
    return store.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]


def forget_conversations(cfg: Config, store: Store) -> list[str]:
    """Forget every conversation: what was removed."""
    n = conversations(store)
    db = store.db
    db.execute("DELETE FROM sessions")
    db.execute("DELETE FROM threads")  # (Telegram threads bound to a conversation; projects' topics stay)
    db.execute("DELETE FROM autopilot")
    db.execute("UPDATE messages SET session_id=NULL")  # (replies to job messages still find their job)
    db.commit()
    keys = [k for p in CONVERSATION_KEYS for k in store.keys(p)]
    for k in keys:
        store.unset(k)
    removed = [f"{n} conversation(s) and {len(keys)} entr{'y' if len(keys) == 1 else 'ies'} about them"]
    for p in conversation_files(cfg):
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
        removed.append(str(p))
    return removed


def size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        if p.is_symlink():
            continue
        if p.is_file():
            total += p.stat().st_size
        else:
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    return total
