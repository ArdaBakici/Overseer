"""The HPC bot's folder in your Obsidian vault (obsidian.folder), e.g. Vault/HPC:

    HPC/General.md                      the research log of work outside projects
    HPC/<project>/Research log.md       a project's research log - written by the agent (a note: edit it too)
    HPC/<project>/Findings.md           its insights and dead ends  } written by Overseer from its records,
    HPC/<project>/Results.md            its results, with graphs     } rewritten when they change
    HPC/<project>/figures/              the graphs of its finished jobs (and figures the agent copies there)

A project's research log used to live on the HPC (RESEARCH_LOG.md in its folder): the first time it's needed, that
file is copied into the vault, which keeps it from then on. The agent's sandbox can read and write this folder and
nothing else of the vault; the assistant (if it shares the vault) sees it but can't change it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from ..core.store import ResultRecord
from . import prompts as P
from .results import FINDING_KINDS, VERDICTS, _metrics, metric_columns
from .ssh import SSHError

log = logging.getLogger(__name__)

LOG_NOTE = "Research log.md"
GENERAL_NOTE = "General.md"
FINDINGS_NOTE = "Findings.md"
RESULTS_NOTE = "Results.md"
FIGURES = "figures"
SYNC_SECONDS = 60
IMPORTED_KEY = "obsidian_imported:"  # + project: its RESEARCH_LOG.md from the HPC was copied in (or there was none)
GENERATED = ("> [!info] Written by Overseer from its records, rewritten when they change - edits here are lost "
             "(the research log is the place for notes).")


def note_name(project: str) -> str:
    """A project's folder name in the vault (its name, minus what file systems and Obsidian don't like)."""
    return re.sub(r'[\\/:*?"<>|#^\[\]]+', "-", project).strip(" .-") or "project"


class VaultFolder:
    def __init__(self, domain):
        self.domain = domain
        self.cfg = domain.cfg
        self.store = domain.store
        self.root: Path = domain.cfg.obsidian_dir

    # ---- paths
    def project_dir(self, project: str | None) -> Path:
        return self.root / note_name(project) if project else self.root

    def log_path(self, project: str | None) -> Path:
        return self.project_dir(project) / LOG_NOTE if project else self.root / GENERAL_NOTE

    # ---- the research log
    async def research_log(self, project: str | None) -> str:
        """The research log for the agent: its beginning (the goal) and its latest entries."""
        if project:
            await self.import_old(project)
        path = self.log_path(project)
        what = "this project" if project else "work outside projects"
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        if not data.strip():
            return f"<research_log path='{path}'>(none yet - start it when you run experiments for {what})</research_log>"
        size = len(data)
        if size <= 12_000:
            body = data.decode(errors="replace")
        else:
            body = (f"{data[:4000].decode(errors='replace').rstrip()}\n[... {size // 1000} KB in all; the latest "
                    f"entries: ...]\n{data[-8000:].decode(errors='replace')}")
        note = (f"\n(The log is {size // 1000} KB - too long: condense the older entries into a short summary.)"
                if size > P.LOG_CONDENSE_BYTES else "")
        return f"<research_log path='{path}'>\n{body.strip()}\n</research_log>{note}"

    async def import_old(self, project: str) -> None:
        """Once: the project's RESEARCH_LOG.md from the HPC into the vault (if the vault has no log yet)."""
        if self.store.get(IMPORTED_KEY + project):
            return
        path = self.log_path(project)
        proj = self.store.get_project(project)
        if path.exists() or not proj or not proj.dir:
            self.store.set(IMPORTED_KEY + project, "no")
            return
        remote = self.domain.remote_path(f"{proj.dir.rstrip('/')}/{P.LOG_FILE}")
        try:
            res = await self.domain.hpc.run(["cat", "--", remote], max_output=4_000_000)
        except SSHError:
            return  # (the HPC is unreachable: next time)
        text = res.stdout if res.ok else ""
        if text.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            head = (f"---\nproject: {project}\nimported_from: \"{proj.dir.rstrip('/')}/{P.LOG_FILE}\"\n---\n"
                    if not text.startswith("---") else "")
            path.write_text(head + text)
            log.info("research log of %s imported from the HPC into %s", project, path)
        self.store.set(IMPORTED_KEY + project, "yes" if text.strip() else "no")

    # ---- what Overseer writes: findings, results, graphs
    def sync(self) -> list[str]:
        """Write the generated notes (and copy the graphs) that changed; the paths written."""
        return self.write(self.records())

    def records(self) -> list[tuple]:
        """What the notes are made of, from the database: [(project, findings, results)]. In the thread that owns
        the database (SQLite refuses others): the event loop's."""
        out = []
        for project in [None, *(p.name for p in self.store.list_projects(include_archived=True))]:
            findings, results = self.store.findings(project), self.store.results(project)
            if findings or results:
                out.append((project, findings, results))
        return out

    def write(self, records: list[tuple]) -> list[str]:
        """The files (any thread): the notes and graphs that changed; the paths written."""
        written = []
        for project, findings, results in records:
            d = self.project_dir(project)
            figures = self._figures(project, results, d)
            for name, text in ((FINDINGS_NOTE, self.findings_note(project, findings)),
                               (RESULTS_NOTE, self.results_note(project, results, figures))):
                if text and _write_if_changed(d / name, text):
                    written.append(str(d / name))
        return written

    def findings_note(self, project: str | None, findings) -> str:
        if not findings:
            return ""
        lines = [_front(project, "findings"), f"# Findings{' - ' + project if project else ''}", "", GENERATED, ""]
        for kind, title in (("insight", "Insights"), ("dead-end", "Dead ends")):
            items = [f for f in findings if f.kind == kind]
            if items:
                lines += [f"## {FINDING_KINDS[kind]} {title}", ""]
                lines += [f"- **F{f.id}** {f.text}" + (f" (job {f.job_id})" if f.job_id else "")
                          + f" - {_day(f.created)}" for f in items]
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def results_note(self, project: str | None, results: list[ResultRecord], figures: dict[str, list[str]]) -> str:
        if not results:
            return ""
        newest = sorted(results, key=lambda r: r.ended or r.created, reverse=True)
        cols = metric_columns(results)
        head = ["result", "name", "state", "verdict", *cols, "conclusion"]
        lines = [_front(project, "results"), f"# Results{' - ' + project if project else ''}", "", GENERATED, "",
                 "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in newest:
            m = _metrics(r)
            cells = [f"[[#{_anchor(r)}|{r.id}]]", r.name, r.state, _verdict(r), *[m.get(c, "") for c in cols],
                     r.conclusion]
            lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
        lines.append("")
        for r in newest:
            lines += [f"## {_anchor(r)}", ""]
            facts = [f"**{k}:** {v}" for k, v in (("state", r.state), ("verdict", _verdict(r)),
                                                  ("runtime", r.elapsed), ("ended", _day(r.ended, True))) if v]
            lines += [" · ".join(facts), ""] if facts else []
            for k in ("hypothesis", "conclusion"):
                if getattr(r, k):
                    lines += [f"**{k.capitalize()}:** {getattr(r, k)}", ""]
            m = _metrics(r)
            if m:
                lines += [", ".join(f"{k} = {v}" for k, v in list(m.items())[:12]), ""]
            lines += [f"![[{name}]]" for name in figures.get(r.id, [])]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _figures(self, project: str | None, results: list[ResultRecord], d: Path) -> dict[str, list[str]]:
        """The graphs of finished jobs (Overseer's job-card graphs) into figures/: {result id: [file names]}."""
        out: dict[str, list[str]] = {}
        limit = self.cfg.obsidian.max_figure_mb * 1024 * 1024
        for r in results:
            if not r.job_id or not r.ended:  # (only final graphs: not every hourly one of a running job)
                continue
            cache = self.cfg.local.state("plots", r.job_id)
            if not cache.is_dir():
                continue
            for f in sorted(cache.glob("*.png")):
                if f.stat().st_size > limit:
                    continue
                name = f"{r.job_id}-{f.name.split('-', 1)[-1]}"
                if _write_if_changed(d / FIGURES / name, f.read_bytes()):
                    log.info("graph %s into the vault", name)
                out.setdefault(r.id, []).append(name)
        return out

    async def loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.write, self.records())  # (reading here, writing in a thread)
            except Exception:
                log.exception("obsidian notes")
            await asyncio.sleep(SYNC_SECONDS)


def _write_if_changed(path: Path, content: str | bytes) -> bool:
    data = content.encode() if isinstance(content, str) else content
    try:
        if path.read_bytes() == data:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{hashlib.sha1(data).hexdigest()[:8]}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def _front(project: str | None, kind: str) -> str:
    return f"---\nproject: {project or 'none'}\ngenerated: overseer\ntags: [hpc, {kind}]\n---"


def _day(ts: float, with_time: bool = False) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d") if ts else ""


def _verdict(r: ResultRecord) -> str:
    return f"{VERDICTS.get(r.verdict, '')} {r.verdict}".strip() if r.verdict else ""


def _anchor(r: ResultRecord) -> str:
    """The heading of a result's section (also its link target in the table)."""
    return re.sub(r"[#^\[\]|]", " ", f"{r.id} {r.name}".strip())[:80].strip()


def _cell(v) -> str:
    """A table cell: one line, and | escaped (Obsidian's tables read \\| as a |, in links too)."""
    return " ".join(str(v or "").split())[:120].replace("|", "\\|")

