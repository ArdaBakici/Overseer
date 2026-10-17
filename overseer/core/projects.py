"""Projects: separate contexts (e.g. dreamer, chemistry), each with its own Telegram topic.

A project has a folder on the HPC, standing instructions for the agent, its own conversations,
jobs, research log and agent settings (backend / model / effort, copied from the main chat when the
project is created). Projects come from config.yaml (`projects:`) or `/project new` in Telegram.

* Jobs the agent submits in a project's topic belong to that project.
* Jobs you submit by hand belong to the project whose folder contains their working directory
  (the deepest one if folders nest); others go to the main chat. `/assign` moves a job.
* In the project's folder, OVERSEER.md holds durable notes the agent reads at the start of each
  conversation and keeps up to date.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .config import BACKENDS, PROJECT_NAME_RE, RESERVED_PROJECT_NAMES, Config
from .store import ProjectRecord, Store

if TYPE_CHECKING:
    from .agent import Agent

log = logging.getLogger(__name__)


class ProjectError(ValueError):
    pass


def check_name(name: str) -> str:
    name = name.strip().lower()
    if not PROJECT_NAME_RE.match(name) or name in RESERVED_PROJECT_NAMES:
        raise ProjectError(f"{name!r} can't be a project name: use up to 32 lowercase letters, digits, - and _ "
                           "(not main/all/general/none)")
    return name


def check_dir(path: str) -> str:
    path = path.strip()
    if not path.startswith(("/", "~/")) or any(c.isspace() for c in path):
        raise ProjectError(f"{path!r}: give the project's folder on the HPC as an absolute path (or ~/...)")
    return path.rstrip("/") or "/"


class Projects:
    def __init__(self, cfg: Config, store: Store, agent: Agent):
        self.cfg = cfg
        self.store = store
        self.agent = agent
        self.notifier = None  # set by the app: creates and closes Telegram topics

    def get(self, name: str | None) -> ProjectRecord | None:
        return self.store.get_project(name) if name else None

    def active(self) -> list[ProjectRecord]:
        return self.store.list_projects()

    def create(self, name: str, dir: str, context: str = "", source: str = "telegram",
               backend: str | None = None, model: str | None = None, effort: str | None = None) -> ProjectRecord:
        """Create a project (or restore an archived one). Its agent settings start as the main chat's."""
        name, dir = check_name(name), check_dir(dir)
        if backend and backend not in BACKENDS:
            raise ProjectError(f"backend must be one of {', '.join(BACKENDS)}")
        p = self.store.get_project(name)
        if p and not p.archived:
            raise ProjectError(f"project {name!r} already exists (folder {p.dir})")
        if p is None:
            p = ProjectRecord(name=name, created=time.time())
        p.dir, p.context, p.archived, p.source = dir, context.strip() or p.context, 0, source
        self.store.upsert_project(p)
        self.agent.init_project_settings(name, backend, model, effort)
        return p

    def archive(self, name: str) -> ProjectRecord:
        p = self.store.get_project(name)
        if not p or p.archived:
            raise ProjectError(f"no active project {name!r}")
        p.archived = 1
        self.store.upsert_project(p)
        if self.agent.autopilot:  # an archived project doesn't keep experimenting
            sessions = [sess.id for sess in self.store.list_sessions(10_000) if sess.project == name]
            if sessions:
                self.agent.autopilot.cancel_all(sessions)
        return p

    def update(self, name: str, *, dir: str | None = None, context: str | None = None) -> ProjectRecord:
        p = self.store.get_project(name)
        if not p:
            raise ProjectError(f"no project {name!r}")
        if dir is not None:
            p.dir = check_dir(dir)
        if context is not None:
            p.context = context.strip()
        self.store.upsert_project(p)
        return p

    def sync_config(self) -> list[str]:
        """Create or update the projects in config.yaml. Folder and context follow the config; the
        agent settings given there apply when they change (Telegram's /model etc. last until then)."""
        created = []
        for name, pc in self.cfg.projects.items():
            p = self.store.get_project(name)
            if p is None or p.archived:
                self.create(name, pc.dir, pc.context, "config", pc.backend, pc.model, pc.effort)
                created.append(name)
            else:
                p.dir, p.source = check_dir(pc.dir), "config"
                p.context = pc.context.strip() or p.context
                self.store.upsert_project(p)
            for field, value in (("backend", pc.backend), ("model", pc.model), ("effort", pc.effort)):
                key = f"cfgsync:{name}:{field}"
                if (value or "") == (self.store.get(key) or ""):
                    continue
                self.store.set(key, value or "")
                if value and name not in created:
                    try:
                        if field == "backend":
                            self.agent.set_backend(value, name)
                        elif field == "model":
                            self.agent.set_model(value, project=name)
                        else:
                            self.agent.set_effort(value, project=name)
                    except ValueError as e:
                        log.warning("projects.%s.%s: %s", name, field, e)
        return created

    def assign_by_dir(self, workdir: str) -> str | None:
        p = self.store.project_for_dir(workdir) if workdir else None
        return p.name if p else None
