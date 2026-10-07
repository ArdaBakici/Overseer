"""Per-session record of what the agent changed on the HPC, with local backups for diff/revert.

Before the agent first modifies a file, its original content is saved on the local machine
(in ``local.state_dir``). That gives the user an exact diff of every change and a one-tap revert. Copies are recorded so revert can remove them.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .hpc import HPC


@dataclass
class Entry:
    path: str  # resolved remote path
    kind: str  # "file" (modified/created/deleted file) | "copy" (created by copy_path)
    existed: bool = False
    backup: str = ""  # local backup filename (kind == "file" and existed)
    source: str = ""  # for copies


class ChangeSet:
    def __init__(self, local_dir: Path):
        self.dir = local_dir
        self.entries: dict[str, Entry] = {}
        index = self.dir / "index.json"
        if index.exists():
            for d in json.loads(index.read_text()):
                self.entries[d["path"]] = Entry(**d)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def _save(self) -> None:
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.dir / "index.json").write_text(json.dumps([asdict(e) for e in self.entries.values()], indent=1))

    def covered(self, path: str) -> bool:
        if path in self.entries:
            return True
        return any(e.kind == "copy" and path.startswith(e.path.rstrip("/") + "/") for e in self.entries.values())

    def record_file(self, path: str, original: str | None) -> None:
        if path in self.entries:
            return
        e = Entry(path=path, kind="file", existed=original is not None)
        if original is not None:
            self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            e.backup = hashlib.sha1(path.encode()).hexdigest() + ".orig"
            fd = os.open(self.dir / e.backup, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(original)
        self.entries[path] = e
        self._save()

    def record_copy(self, dest: str, source: str) -> None:
        self.entries[dest] = Entry(path=dest, kind="copy", source=source)
        self._save()

    def original(self, path: str) -> str | None:
        e = self.entries[path]
        if not e.existed:
            return None
        return (self.dir / e.backup).read_text()

    def summary_lines(self) -> list[str]:
        out = []
        for e in self.entries.values():
            if e.kind == "copy":
                out.append(f"copied {e.source} → {e.path}")
            else:
                out.append(("modified " if e.existed else "created ") + e.path)
        return out

    async def diff(self, hpc: HPC) -> str:
        parts = []
        copies = [e for e in self.entries.values() if e.kind == "copy"]
        for e in copies:
            parts.append(f"# copied {e.source} -> {e.path}\n")
        for e in self.entries.values():
            if e.kind != "file":
                continue
            before = self.original(e.path)
            after = await hpc.read_for_edit(e.path) if await hpc.exists(e.path) else None
            if before == after:
                continue
            parts.append("".join(difflib.unified_diff(
                (before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True),
                fromfile="/dev/null" if before is None else f"a{e.path}",
                tofile="/dev/null" if after is None else f"b{e.path}",
            )))
        return "\n".join(p if p.endswith("\n") else p + "\n" for p in parts if p)

    async def revert(self, hpc: HPC) -> list[str]:
        """Restore originals, remove created files and copies. Returns a log of what was done."""
        done = []
        for e in reversed(list(self.entries.values())):
            if e.kind == "file":
                original = self.original(e.path)
                if original is None:
                    if await hpc.exists(e.path):
                        await hpc.delete(e.path)
                        done.append(f"removed {e.path}")
                else:
                    await hpc.write_file(e.path, original)
                    done.append(f"restored {e.path}")
            else:
                if await hpc.exists(e.path):
                    await hpc.delete(e.path)
                    done.append(f"removed copy {e.path}")
        self.entries.clear()
        self._save()
        return done
