"""Graphs of a job, from a plotting command the agent wrote (`ov plots set`): no model call per graph.

The agent writes a small script in the project (reading the job's metrics files or log) and registers the command
that runs it, for one job, a sweep (a name pattern) or a project, like `ov metrics`. Overseer runs it on the HPC -
when the job ends, every hour while it runs, and when the 📈 button is pressed - with an empty folder for its
output, and posts the PNGs it wrote under the job's card (edited in place when they're refreshed).

Placeholders: {job_id} {name} {workdir} {stdout} {project} (the project's folder) {out} (where to write PNGs).
"""

from __future__ import annotations

import base64
import fnmatch
import shlex
import time

from ..core.store import JobRecord, MetricSpec, Store

PLOT_TIMEOUT = 90  # seconds: plotting from logs is quick; heavy work belongs in the script's own job
MAX_IMAGES = 4
MAX_BYTES = 10 * 1024 * 1024  # Telegram's limit for photos sent by bots


def resolve_plot(store: Store, rec: JobRecord) -> MetricSpec | None:
    """The plotting command for a job: its own, else the newest pattern matching its name, else its project's."""
    spec = store.get_metric_spec("job", rec.job_id)
    if spec and spec.plot:
        return spec
    for p in store.metric_specs("pattern"):  # newest first
        if p.plot and rec.name and fnmatch.fnmatch(rec.name, p.key):
            return p
    spec = store.get_metric_spec("project", rec.project) if rec.project else None
    return spec if spec and spec.plot else None


def fill(command: str, rec: JobRecord, out: str, project_dir: str) -> str:
    q = shlex.quote
    return (command.replace("{job_id}", q(rec.job_id)).replace("{name}", q(rec.name or ""))
            .replace("{workdir}", q(rec.workdir or ".")).replace("{stdout}", q(rec.stdout or "/dev/null"))
            .replace("{project}", q(project_dir or rec.workdir or ".")).replace("{out}", q(out)))


async def run_plots(hpc, store: Store, rec: JobRecord, command: str) -> tuple[list[tuple[str, bytes]], str]:
    """Run the command on the HPC: ([(file name, PNG bytes)], what went wrong - "" if it worked)."""
    q = shlex.quote
    proj = store.get_project(rec.project) if rec.project else None
    project_dir = proj.dir.replace("~", hpc.home, 1) if proj and proj.dir.startswith("~") else (proj.dir if proj else "")
    out = f"{hpc.home or '/tmp'}/.overseer/plots/{rec.job_id}-{int(time.time() * 1000)}"
    script = (f"mkdir -p {q(out)} && cd {q(rec.workdir or '.')} 2>/dev/null; "
              f"nice -n 10 timeout {PLOT_TIMEOUT} bash -c {q(fill(command, rec, out, project_dir))} >&2; "
              f"echo __RC__$?; for f in {q(out)}/*.png; do [ -f \"$f\" ] && echo \"__PNG__$(stat -c %s \"$f\") $f\"; "
              "done")
    try:
        res = await hpc.run(hpc._slurm(["bash", "-c", script]), timeout=PLOT_TIMEOUT + 30, max_output=200_000)
        rc, pngs = None, []
        for line in res.stdout.splitlines():
            if line.startswith("__RC__"):
                rc = line[6:].strip()
            elif line.startswith("__PNG__"):
                size, _, path = line[7:].partition(" ")
                pngs.append((int(size) if size.isdigit() else 0, path))
        images, skipped = [], []
        for size, path in sorted(pngs, key=lambda p: p[1])[:MAX_IMAGES]:
            name = path.rsplit("/", 1)[-1]
            if size > MAX_BYTES:
                skipped.append(f"{name} is over 10 MB")
                continue
            data = await hpc.run(["base64", "-w0", "--", path], timeout=60, max_output=int(size * 1.4) + 1000)
            try:
                images.append((name, base64.b64decode(data.stdout.strip(), validate=False)))
            except ValueError:
                skipped.append(f"{name} couldn't be read")
        err = ""
        if rc not in (None, "0"):
            why = "it took over 90 s and was stopped" if rc == "124" else f"exit {rc}"
            err = f"the plotting command failed ({why}): {res.stderr.strip()[-800:]}"
        elif not images:
            err = f"the plotting command wrote no PNG into {{out}}. {res.stderr.strip()[-400:]}".strip()
        if len(pngs) > MAX_IMAGES:
            skipped.append(f"only the first {MAX_IMAGES} of {len(pngs)} images are shown")
        if skipped:
            err = (err + "; " if err else "") + "; ".join(skipped)
        return images, err
    finally:
        try:
            await hpc.run(["rm", "-rf", "--", out], timeout=30)
        except Exception:
            pass
