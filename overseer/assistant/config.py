"""The personal assistant's configuration: its `vault` section of config.yaml.

The assistant runs as its own bot (domain: assistant) with its own config file, Telegram token (telegram.token_env,
from the .env file), state folder and tmp folder - never the HPC bot's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.config import ConfigError


@dataclass
class VaultConfig:
    """The Obsidian vault: the assistant's memory. Overseer keeps a history of every change in its own state folder
    (a git repository outside the vault, so nothing extra is synced), and each turn's changes can be undone."""

    path: str = ""  # the vault's folder on this machine (the one Syncthing keeps in sync), e.g. /home/me/Vault
    assistant_dir: str = "Assistant"  # the assistant's own notes, inside the vault: persona, about you, memory
    tasks_file: str = ""  # where new tasks go ("" = <assistant_dir>/Tasks.md); tasks are read from the whole vault
    daily_folder: str | None = None  # daily notes; None = Obsidian's own setting (.obsidian/daily-notes.json)
    daily_format: str | None = None  # e.g. YYYY-MM-DD (Obsidian's format); None = Obsidian's setting
    history: bool = True  # keep a history of the vault's changes (git, in the state folder) and offer Undo


@dataclass
class MediaConfig:
    """Pictures, GIFs and videos from the internet for the assistant's replies (its shell has no internet: Overseer
    downloads them, from public addresses only, checking what the file is)."""

    fetch: bool = True  # `ov fetch <url>`: download a picture, GIF or video into the reply
    max_mb: int = 20  # the largest file it fetches
    # `ov gif <query>`: Giphy's search. The name of the variable holding your Giphy API key in .env (e.g.
    # GIPHY_API_KEY); "" = off.
    giphy_key_env: str = ""
    giphy_rating: str = "pg-13"  # g | pg | pg-13 | r


SECTIONS = {"vault": VaultConfig, "media": MediaConfig}


def _vault_dir(cfg) -> Path:
    return Path(cfg.vault.path).expanduser().resolve()


DERIVED = {
    "vault_dir": _vault_dir,
    "vault_git_dir": lambda cfg: cfg.local.state("vault.git"),
}


def validate(cfg) -> None:
    if not cfg.vault.path:
        raise ConfigError("vault.path is required: your Obsidian vault's folder on this machine")
    vault = Path(cfg.vault.path).expanduser()
    if not vault.is_absolute():
        raise ConfigError("vault.path must be an absolute path")
    if not vault.is_dir():
        raise ConfigError(f"vault.path {vault} isn't a folder")
    base = Path(cfg.local.base_dir or ".").resolve()
    vault = vault.resolve()
    if vault in (Path("/"), Path.home()) or base == vault or base.is_relative_to(vault) or vault.is_relative_to(base):
        raise ConfigError(f"vault.path {vault}: point it at the vault itself, not at Overseer's folder or above it")
    # Its own state and tmp folders: the HPC bot's (./state, /tmp/overseer) hold its SSH key and sockets.
    if cfg.local._resolve(cfg.local.state_dir) == base / "state":
        raise ConfigError("local.state_dir: give the assistant its own, e.g. ./state/assistant (./state is the HPC "
                          "bot's)")
    if cfg.local._resolve(cfg.local.tmp_dir) == Path("/tmp/overseer").resolve():
        raise ConfigError("local.tmp_dir: give the assistant its own, e.g. /tmp/overseer-assistant")
    if not 1 <= cfg.media.max_mb <= 50:
        raise ConfigError("media.max_mb: 1 to 50 (Telegram's limit for files bots send)")
    if cfg.media.giphy_rating not in ("g", "pg", "pg-13", "r"):
        raise ConfigError("media.giphy_rating: g, pg, pg-13 or r")
    if cfg.telegram.token_env == "TELEGRAM_BOT_TOKEN":
        raise ConfigError("telegram.token_env: the assistant is its own bot - name its token, e.g. "
                          "ASSISTANT_BOT_TOKEN (in the .env file)")
    # Its own logins and sessions for the agent CLIs, unless set (`overseer login -c <this config>`): the HPC bot's
    # hold its skills and conversations.
    for backend, name in ((cfg.agent.claude_code, "claude"), (cfg.agent.codex, "codex")):
        if cfg.local._resolve(backend.home) == base / "state" / name:
            backend.home = str(cfg.local.state(name))
