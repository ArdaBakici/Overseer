"""Skills as addons: know-how an agent wrote down, installed for agents once the researcher approves it.

Many limitations agents run into aren't missing features but missing know-how: "on this cluster, jobs over 24 h
need --qos long", "the eval script needs --no-cache on the shared filesystem". An agent writes a skill (a SKILL.md
with a name and a description) and proposes it: `ov skill propose skill.md [--all]`. The researcher approves or
declines it in Telegram. Approved skills are installed into every run of the project (or of every conversation, with
--all), where both CLIs find them: Claude Code in `.claude/skills/`, Codex in `.agents/skills/` of the run's folder.

A skill only adds knowledge: its frontmatter may hold nothing but a name and a description (no allowed-tools or other
fields that could change what an agent may do), and agents keep running inside their sandboxes. Overseer writes the
files itself, into a folder the agent can write too, so it never follows links there.
"""

from __future__ import annotations

import json
import os
import re
import stat

import yaml

from .notify import Notifier, Outgoing, esc
from .store import SkillRecord, Store

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
RESERVED = {"overseer"}
MAX_BYTES = 30_000
MARKER = ".overseer-addon"  # in each installed skill's folder: removed again when the skill is


class SkillError(ValueError):
    pass


def parse_skill(text: str) -> tuple[str, str]:
    """A SKILL.md's (name, description), or SkillError saying what's wrong with it."""
    if len(text.encode()) > MAX_BYTES:
        raise SkillError(f"a skill must stay under {MAX_BYTES // 1000} KB: keep it to what an agent needs to know")
    m = re.match(r"^---\s*\n(.*?)\n---\s*(\n|$)", text, re.S)
    if not m:
        raise SkillError("a skill starts with frontmatter: ---, name: <name>, description: <when to use it>, ---")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"the frontmatter isn't valid YAML: {e}") from e
    if not isinstance(meta, dict):
        raise SkillError("the frontmatter must be name: and description: lines")
    extra = set(meta) - {"name", "description"}
    if extra:
        raise SkillError(f"only name and description are allowed in the frontmatter, not {', '.join(sorted(extra))}: "
                         "other fields could change what agents may do")
    name, desc = str(meta.get("name") or ""), " ".join(str(meta.get("description") or "").split())
    if not NAME_RE.match(name):
        raise SkillError(f"skill name {name!r}: lowercase letters, digits and -, e.g. cluster-qos")
    if name in RESERVED:
        raise SkillError(f"{name!r} is Overseer's own skill: pick another name")
    if len(desc) < 20 or len(desc) > 1024:
        raise SkillError("the description (20-1024 characters) says what the skill is for and when to use it")
    if not text[m.end():].strip():
        raise SkillError("the skill has no body below its frontmatter")
    return name, desc


# ------------------------------------------------------------------ writing into a run's folder
def _open_dir(parent_fd: int, name: str, create: bool = True) -> int:
    """A sub-folder, opened without following links (one that is a link, or a file, is replaced)."""
    try:
        st = os.lstat(name, dir_fd=parent_fd)
        if not stat.S_ISDIR(st.st_mode):
            os.unlink(name, dir_fd=parent_fd)
            raise FileNotFoundError
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)


def _write(dir_fd: int, name: str, data: bytes) -> None:
    try:
        if not stat.S_ISREG(os.lstat(name, dir_fd=dir_fd).st_mode):
            os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _remove_tree(parent_fd: int, name: str) -> None:
    """Remove a folder Overseer made (its files only; links are unlinked, never followed)."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError:
        return
    try:
        for entry in os.listdir(fd):
            if stat.S_ISDIR(os.lstat(entry, dir_fd=fd).st_mode):
                _remove_tree(fd, entry)
            else:
                os.unlink(entry, dir_fd=fd)
    finally:
        os.close(fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError:
        pass


def write_skills(root, parts: tuple[str, ...], skills: dict[str, str], prune: bool = True) -> None:
    """Put skills (name -> SKILL.md text) into root/<parts>/<name>/SKILL.md; with `prune`, remove the addon skills
    installed before that aren't in `skills` any more. Links under `root` are never followed."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts:
            nfd = _open_dir(fd, part)
            os.close(fd)
            fd = nfd
        if prune:
            for entry in os.listdir(fd):
                if entry in skills:
                    continue
                try:
                    sub = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except OSError:
                    continue
                try:
                    ours = MARKER in os.listdir(sub)
                finally:
                    os.close(sub)
                if ours:
                    _remove_tree(fd, entry)
        for name, text in skills.items():
            sfd = _open_dir(fd, name)
            try:
                _write(sfd, "SKILL.md", text.encode())
                if prune:
                    _write(sfd, MARKER, b"installed by Overseer: approved skill (addons.py)\n")
            finally:
                os.close(sfd)
    finally:
        os.close(fd)


SKILL_DIRS = {"claude_code": (".claude", "skills"), "codex": (".agents", "skills")}


class Addons:
    def __init__(self, store: Store, agent, notifier: Notifier):
        self.store = store
        self.agent = agent
        self.notifier = notifier
        agent.addons = self

    def for_project(self, project: str | None) -> list[SkillRecord]:
        """The skills a conversation gets: every agent's, then its project's (which win over a global one of the same
        name)."""
        out: dict[str, SkillRecord] = {}
        for k in self.store.skills():
            if k.project == "":
                out.setdefault(k.name, k)
        for k in self.store.skills():
            if project and k.project == project:
                out[k.name] = k
        return list(out.values())

    def install(self, s, workdir, backend: str) -> None:
        """Before a run: the approved skills into its folder, where the CLI looks for them."""
        write_skills(workdir, SKILL_DIRS.get(backend, (".agents", "skills")),
                     {k.name: k.content for k in self.for_project(s.project)})

    # ------------------------------------------------------------------ proposals
    async def propose(self, s, text: str, everyone: bool = False) -> SkillRecord:
        name, desc = parse_skill(text)
        project = "" if everyone or not s.project else s.project
        k = self.store.add_skill(SkillRecord(name=name, description=desc, content=text.strip() + "\n",
                                             project=project, session_id=s.id))
        msg = Outgoing(self.card_text(k), session_id=s.id,
                       buttons=[[("✅ Install", f"sk:{k.id}"), ("✖ No", f"skx:{k.id}")]])
        if hasattr(self.notifier, "post"):
            k.refs = json.dumps(await self.notifier.post(msg))
            self.store.save_skill(k)
        else:
            await self.notifier.send(msg)
        return k

    def _scope(self, k: SkillRecord) -> str:
        return f"project {k.project}" if k.project else "every agent"

    def card_text(self, k: SkillRecord) -> str:
        current = [x for x in self.store.skills() if x.name == k.name and x.project == k.project]
        body = k.content.split("---", 2)[-1].strip()
        preview = body if len(body) <= 2200 else body[:2200] + "\n…"
        return (f"🧩 <b>Skill proposal: {esc(k.name)}</b> · for {esc(self._scope(k))}"
                + (" (replaces the installed one)" if current else "")
                + f"\n<i>{esc(k.description)}</i>\n<blockquote expandable>{esc(preview)}</blockquote>")

    async def _answer(self, k: SkillRecord, text: str) -> None:
        refs = json.loads(k.refs or "{}")
        if refs and hasattr(self.notifier, "edit"):
            await self.notifier.edit(refs, self.card_text(k).split("\n<blockquote")[0] + f"\n\n{text}")

    async def approve(self, sid: int) -> str:
        k = self.store.get_skill(sid)
        if not k or k.status != "proposed":
            return "This skill proposal was already answered."
        for old in self.store.skills():
            if old.name == k.name and old.project == k.project:
                old.status = "replaced"
                self.store.save_skill(old)
        k.status = "installed"
        self.store.save_skill(k)
        if k.session_id:
            self.agent.add_note(k.session_id, f"The researcher installed your skill {k.name} for {self._scope(k)}: "
                                              "agents get it from their next run.")
        text = f"✅ <b>Installed</b> for {esc(self._scope(k))}: agents get it from their next run. (/skills)"
        await self._answer(k, text)
        return text

    async def decline(self, sid: int) -> str:
        k = self.store.get_skill(sid)
        if not k or k.status != "proposed":
            return "This skill proposal was already answered."
        k.status = "declined"
        self.store.save_skill(k)
        if k.session_id:
            self.agent.add_note(k.session_id, f"The researcher declined your skill {k.name}.")
        await self._answer(k, "✖ <b>Declined.</b>")
        return "✖ Declined."

    def remove(self, sid: int) -> SkillRecord | None:
        k = self.store.get_skill(sid)
        if not k or k.status != "installed":
            return None
        k.status = "removed"
        self.store.save_skill(k)
        return k
