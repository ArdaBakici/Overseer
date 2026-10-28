"""`overseer tui`: every bot of this Overseer folder, live, in the terminal. Read-only: it shows what the bots are doing
and never touches their conversations, Telegram or the HPC. Its only actions start and stop the bots themselves
(in the background: they keep running after the TUI closes).

Keys: s start · x stop · r restart · 1-6 tabs · q quit.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Header, Label, ListItem, ListView, RichLog, Static,
                             TabbedContent, TabPane)

from ..core import daemon
from ..core.live import Style, render_event
from . import data as D

REFRESH = 2.0  # seconds
NAMES = {"claude_code": "Claude Code", "codex": "Codex"}
ORIGINS = {"user": "you", "wakeup": "wake-up", "checkin": "check-in", "compact": "compaction",
           "subagent": "subagent", "routine": "routine"}


class Confirm(ModalScreen[bool]):
    DEFAULT_CSS = """
    Confirm { align: center middle; }
    #box { width: 64; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    #buttons { height: auto; margin-top: 1; }
    #buttons Button { margin-right: 2; }
    """
    BINDINGS = [Binding("escape", "no", "cancel")]

    def __init__(self, question: str, yes: str = "Yes"):
        super().__init__()
        self.question, self.yes = question, yes

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(self.question)
            with Horizontal(id="buttons"):
                yield Button(self.yes, variant="warning", id="yes")
                yield Button("Cancel", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_no(self) -> None:
        self.dismiss(False)


class Monitor(App):
    TITLE = "Overseer"
    CSS = """
    #bots { width: 30; border-right: solid $primary-background; }
    #bots ListItem { padding: 0 1; }
    #summary { height: auto; padding: 0 1; background: $boost; }
    .table-title { padding: 1 1 0 1; text-style: bold; }
    #domain DataTable { height: auto; max-height: 20; }
    #details { height: auto; max-height: 14; padding: 0 1; border-top: solid $primary-background; }
    #conversations { width: 46%; }
    #transcript { border-left: solid $primary-background; }
    #overview, #usage { padding: 0 1; }
    """
    BINDINGS = [
        Binding("s", "start", "start"), Binding("x", "stop", "stop"), Binding("r", "restart", "restart"),
        Binding("1", "tab('overview')", "overview", show=False), Binding("2", "tab('domain')", show=False),
        Binding("3", "tab('agents')", show=False), Binding("4", "tab('wakeups')", show=False),
        Binding("5", "tab('usage')", show=False), Binding("6", "tab('log')", show=False),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, base: Path):
        super().__init__()
        self.base = base
        self.bots: list[D.Bot] = D.discover(base)
        self.selected = 0
        self.busy: set[str] = set()  # bots being started or stopped
        self.conversation: str | None = None
        self._shown: dict[str, object] = {}  # what each pane last showed (redrawn only when it changed)

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield ListView(*[ListItem(Label(self._bot_line(b)), id=f"bot{i}") for i, b in enumerate(self.bots)],
                           id="bots")
            with Vertical():
                yield Static(id="summary")
                with TabbedContent(id="tabs", initial="overview"):
                    with TabPane("Overview", id="overview"):
                        yield VerticalScroll(Static(id="overview-body"))
                    with TabPane("Bot", id="domain"):
                        yield VerticalScroll(id="domain-body")
                        yield Static(id="details")
                    with TabPane("Agents", id="agents"):
                        with Horizontal():
                            yield DataTable(id="conversations", cursor_type="row", zebra_stripes=True)
                            yield RichLog(id="transcript", wrap=True, markup=False, highlight=False)
                    with TabPane("Wake-ups", id="wakeups"):
                        yield DataTable(id="wakeup-table", cursor_type="row", zebra_stripes=True)
                    with TabPane("Usage", id="usage"):
                        yield VerticalScroll(Static(id="usage-body"))
                    with TabPane("Log", id="log"):
                        yield RichLog(id="log-body", wrap=False, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#conversations", DataTable).add_columns("", "conversation", "project", "agent", "last")
        self.query_one("#wakeup-table", DataTable).add_columns("#", "conversation", "what", "when", "repeats")
        self.sub_title = str(self.base)
        self.refresh_all()
        self.set_interval(REFRESH, self.refresh_all)

    # ------------------------------------------------------------------ helpers
    @property
    def bot(self) -> D.Bot | None:
        return self.bots[self.selected] if self.bots else None

    def _bot_line(self, bot: D.Bot, h: D.Health | None = None) -> str:
        if bot.error:
            return f"{bot.label}\n  ⚠ config error"
        h = h or D.health(bot)
        state = "… working" if str(bot.config) in self.busy else h.label
        extra = "\n  ↻ restarts after a crash" if h.running and h.watched else ""
        return f"{bot.label}\n  {state}{extra}"

    def _changed(self, key: str, value) -> bool:
        if self._shown.get(key) == value:
            return False
        self._shown[key] = value
        return True

    # ------------------------------------------------------------------ refresh
    def refresh_all(self) -> None:
        for i, b in enumerate(self.bots):
            line = self._bot_line(b)
            if self._changed(f"bot{i}", line):
                self.query_one(f"#bot{i}", ListItem).query_one(Label).update(line)
        bot = self.bot
        if not bot:
            self.query_one("#summary", Static).update(
                "No bot is set up in this folder yet: run [b]uv run overseer setup[/b].")
            return
        h = D.health(bot)
        self.query_one("#summary", Static).update(self._summary(bot, h))
        con = D.connect_ro(bot.cfg.local.db_path) if bot.cfg else None
        try:
            active = self.query_one("#tabs", TabbedContent).active
            if active == "overview":
                self.query_one("#overview-body", Static).update(self._overview(bot, h))
            if active == "domain":
                self._domain(h)
            if active == "agents":
                self._agents(bot, h, con)
            if active == "wakeups":
                self._wakeups(con)
            if active == "usage":
                self.query_one("#usage-body", Static).update(self._usage(con))
            if active == "log":
                self._log(bot)
            self._domain_label(h)
        finally:
            if con:
                con.close()

    @on(TabbedContent.TabActivated)
    def tab_changed(self) -> None:
        self._shown = {k: v for k, v in self._shown.items() if k.startswith("bot")}
        self.refresh_all()

    @on(ListView.Highlighted, "#bots")
    def bot_changed(self, event: ListView.Highlighted) -> None:
        if event.item is not None and event.item.id:
            self.selected = int(event.item.id[3:])
            self.conversation = None
            self._shown = {k: v for k, v in self._shown.items() if k.startswith("bot")}
            self.refresh_all()

    def action_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    # ------------------------------------------------------------------ the panes
    def _summary(self, bot: D.Bot, h: D.Health) -> Text:
        if bot.error:
            return Text(f"⚠ {bot.config.name}: {bot.error}", style="bold red")
        st = h.status
        if h.running:
            parts = [f"● running · pid {h.pid}"]
            if st.get("started"):
                parts.append(f"up {D.ago(st['started'])}")
            if h.watched:
                n = sum(1 for t in h.supervisor.get("restarts", []) if t > time.time() - 86400)
                parts.append("↻ watched: restarted after a crash" + (f" ({n}× in the last day)" if n else ""))
            else:
                parts.append("not watched (no restart after a crash)")
            if h.stale:
                parts.append("⚠ no heartbeat for over a minute")
            style = "bold yellow" if h.stale else "bold green"
        elif h.watched:
            parts = [f"↻ the watcher is {h.supervisor.get('state', 'restarting')}",
                     f"last exit code {h.supervisor.get('last_exit')}"]
            style = "bold yellow"
        else:
            last = st.get("stopped") or st.get("updated")
            parts = ["○ stopped" + (f" (last seen {D.ago(last)} ago)" if last else "")]
            gave_up = h.supervisor.get("state") == "gave up"
            parts.append("it kept crashing at start: see the Log tab" if gave_up else "press s to start it")
            style = "bold"
        return Text(f"{bot.label}: " + " · ".join(parts), style=style)

    def _overview(self, bot: D.Bot, h: D.Health) -> Table | Text:
        if bot.error:
            return Text(bot.error)
        st = h.status
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold", no_wrap=True)
        t.add_column()
        if not st:
            t.add_row("status", "never started yet: press s to start it")
        else:
            if not h.running:
                t.add_row("", Text("(as of when it last ran)", style="dim"))
            t.add_row("bot", f"{st.get('bot') or '?'} · Overseer {st.get('version', '?')} · config "
                             f"{Path(st.get('config') or bot.config).name}")
            tg = st.get("telegram") or {}
            t.add_row("Telegram", Text("reachable", style="green") if tg.get("ok") else
                      Text(f"unreachable: {tg.get('waiting', 0)} message(s) waiting", style="red"))
            for label, text, ok in (st.get("domain") or {}).get("lines", []):
                t.add_row(label, Text(str(text), style={True: "green", False: "red"}.get(ok, "")))
            runs = st.get("runs") or []
            if runs:
                for r in runs:
                    who = NAMES.get(r["backend"], r["backend"]) + (f" ({r['account']})" if r.get("account") else "")
                    t.add_row("working" if r is runs[0] else "", f"▶ {r['session']}"
                              + (f" · 📁 {r['project']}" if r.get("project") else "")
                              + f" · {who} · for {ORIGINS.get(r['origin'], r['origin'])} · {D.ago(r['started'])}")
            else:
                t.add_row("working", "no agent run in progress")
            backend = st.get("forced") or st.get("backend") or ""
            t.add_row("agent", NAMES.get(backend, backend) + (" (the other CLI ran out of usage)"
                                                               if st.get("forced") else ""))
            for b, a in (st.get("accounts") or {}).items():
                names = []
                for n in a.get("names", []):
                    mark = "▶ " if n == a.get("active") else ""
                    out = a.get("out", {}).get(n)
                    names.append(f"{mark}{n}" + (f" (out until {time.strftime('%H:%M', time.localtime(out))})"
                                                 if out else ""))
                if b in (st.get("clis") or {}) or len(names) > 1:
                    t.add_row(NAMES.get(b, b), " · ".join(names) + (f"   [{st['clis'][b]}]"
                                                                    if b in (st.get("clis") or {}) else ""))
        note = daemon.linger_note()
        if note:
            t.add_row("note", Text(note, style="yellow"))
        return t

    def _domain_label(self, h: D.Health) -> None:
        tables = (h.status.get("domain") or {}).get("tables") or []
        label = tables[0]["title"] if tables else "Bot"
        if self._changed("domain-label", label):
            self.query_one("#tabs", TabbedContent).get_tab("domain").label = label

    def _domain(self, h: D.Health) -> None:
        tables = (h.status.get("domain") or {}).get("tables") or []
        if self._changed("domain", json_key(tables)):
            self._build_domain(tables)

    @work(exclusive=True, group="domain")
    async def _build_domain(self, tables: list[dict]) -> None:
        body = self.query_one("#domain-body", VerticalScroll)
        await body.remove_children()  # (before the new tables: they reuse the ids)
        self._details: dict[str, list[str]] = {}
        if not tables:
            await body.mount(Static("Nothing yet: the bot shows its jobs (or its day) here once it has run.",
                                    classes="table-title"))
        for i, tb in enumerate(tables):
            dt = DataTable(id=f"dt{i}", cursor_type="row", zebra_stripes=True)
            await body.mount(Label(f"{tb['title']} ({len(tb['rows'])})", classes="table-title"), dt)
            dt.add_columns(*tb["columns"])
            for row in tb["rows"]:
                dt.add_row(*[str(c) for c in row])
            self._details[f"dt{i}"] = tb.get("details") or []

    @on(DataTable.RowHighlighted)
    def row(self, event: DataTable.RowHighlighted) -> None:
        tid = event.data_table.id or ""
        if tid.startswith("dt"):
            details = getattr(self, "_details", {}).get(tid) or []
            text = details[event.cursor_row] if event.cursor_row < len(details) else ""
            self.query_one("#details", Static).update(Text(text))
        elif tid == "conversations":
            key = event.row_key.value if event.row_key else None
            if key and key != self.conversation:
                self.conversation = key
                self._shown.pop("transcript", None)
                self._transcript(self.bot)

    def _agents(self, bot: D.Bot, h: D.Health, con) -> None:
        running = {r["session"] for r in h.status.get("runs") or []} if h.running else set()
        convs = D.conversations(con)
        convs.sort(key=lambda c: (c["id"] not in running, -c["updated"]))
        key = [(c["id"], c["updated"], c["id"] in running) for c in convs]
        if self._changed("conversations", key):
            dt = self.query_one("#conversations", DataTable)
            dt.clear()
            for c in convs:
                name = c["name"] or c["id"]
                kind = {"worker": "🤖 ", "reviewer": "🛠 "}.get(c["role"], "")
                dt.add_row("▶" if c["id"] in running else "", kind + name, c["project"],
                           NAMES.get(c["backend"], c["backend"]), D.ago(c["updated"]), key=c["id"])
            if self.conversation is None and convs:
                self.conversation = convs[0]["id"]
        self._transcript(bot)

    def _transcript(self, bot: D.Bot | None) -> None:
        if not bot or not bot.state_dir or not self.conversation:
            return
        path = bot.state_dir / "transcripts" / f"{self.conversation}.jsonl"
        try:
            stamp = (path.stat().st_size, path.stat().st_mtime)
        except OSError:
            stamp = None
        if not self._changed("transcript", (self.conversation, stamp)):
            return
        log = self.query_one("#transcript", RichLog)
        log.clear()
        if stamp is None:
            log.write(Text("No live log for this conversation (it starts with its next request).", style="dim"))
            return
        st, last_message = Style(True), None
        for ev in D.transcript_events(bot.state_dir, self.conversation):
            for line in render_event(ev, st, last_message=last_message):
                log.write(Text.from_ansi(line))
            if ev.get("kind") in ("message", "turn"):
                last_message = ev.get("text") if ev["kind"] == "message" else None

    def _wakeups(self, con) -> None:
        ws = D.wakeups(con)
        if not self._changed("wakeups", [(w["id"], w["due"]) for w in ws] + [int(time.time() // 30)]):
            return
        dt = self.query_one("#wakeup-table", DataTable)
        dt.clear()
        for w in ws:
            what = {"jobs": f"job(s) {w['jobs']} end", "timer": "timer"}.get(w["kind"], f"{w['kind']} {w['jobs']}".strip())
            if w["note"]:
                what += f": {w['note']}"
            every = f"every {D.ago(time.time() - w['every'])}" if w["every"] else ""
            dt.add_row(str(w["id"]), w["session"] or "", what[:90], D.until(w["due"]) if w["due"] else "when it ends",
                       every)

    def _usage(self, con) -> Table:
        now = time.time()
        lt = time.localtime(now)
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        out = Table(show_edge=False, pad_edge=False, expand=False)
        out.add_column("", style="bold")
        periods = [("today", midnight), ("7 days", now - 7 * 86400), ("30 days", now - 30 * 86400)]
        rows = {name: D.usage(con, since) for name, since in periods}
        for name, _ in periods:
            out.add_column(name, justify="right")

        def cell(rs: list[dict]) -> str:
            tokens = sum((r["input"] or 0) + (r["output"] or 0) for r in rs)
            cost = sum(r["cost"] or 0 for r in rs)
            return (f"{D.human(tokens)} tokens" + (f" · ${cost:.2f}" if cost else "")) if rs else "-"

        out.add_row("all runs", *[cell(rows[n]) for n, _ in periods])
        groups: dict[str, set] = {"by agent": set(), "by account": set(), "for": set()}
        for rs in rows.values():
            for r in rs:
                groups["by agent"].add(r["backend"])
                if r.get("account"):
                    groups["by account"].add(f"{r['backend']}|{r['account']}")
                groups["for"].add(r["origin"])
        for b in sorted(groups["by agent"]):
            out.add_row(NAMES.get(b, b), *[cell([r for r in rows[n] if r["backend"] == b]) for n, _ in periods])
        if len(groups["by account"]) > 1:
            for key in sorted(groups["by account"]):
                b, a = key.split("|")
                out.add_row(f"  {a} ({NAMES.get(b, b)})", *[cell([r for r in rows[n] if r["backend"] == b and
                                                                   r.get("account") == a]) for n, _ in periods])
        for o in sorted(groups["for"]):
            out.add_row(f"for {ORIGINS.get(o, o)}", *[cell([r for r in rows[n] if r["origin"] == o])
                                                      for n, _ in periods])
        return out

    def _log(self, bot: D.Bot) -> None:
        if not bot.state_dir:
            return
        lines = D.log_tail(bot.state_dir)
        if not self._changed("log", (len(lines), lines[-1] if lines else "")):
            return
        log = self.query_one("#log-body", RichLog)
        log.clear()
        if not lines:
            log.write(Text("No log yet: it starts when the bot runs.", style="dim"))
        for line in lines:
            style = "red" if " ERROR " in line or "Traceback" in line else "yellow" if " WARNING " in line else ""
            log.write(Text(line, style=style))

    # ------------------------------------------------------------------ start / stop
    def action_start(self) -> None:
        bot = self.bot
        if not bot or bot.error or str(bot.config) in self.busy:
            return
        if D.health(bot).running:
            self.notify(f"{bot.label} is already running.")
            return
        self._control(bot, "start")

    def action_stop(self) -> None:
        self._ask_stop("stop")

    def action_restart(self) -> None:
        self._ask_stop("restart")

    def _ask_stop(self, what: str) -> None:
        bot = self.bot
        if not bot or bot.error or str(bot.config) in self.busy:
            return
        h = D.health(bot)
        if not h.running and not h.watched:
            if what == "restart":
                self._control(bot, "start")
            else:
                self.notify(f"{bot.label} isn't running.")
            return
        runs = len(h.status.get("runs") or [])
        cut = (f"\n\n{runs} agent run(s) in progress will be cut off; they're offered again with ▶️ Continue at the "
               "next start." if runs else "")
        question = f"{what.capitalize()} {escape(bot.label)}?{cut}"

        def answered(yes: bool | None) -> None:
            if yes:
                self._control(bot, what)
        self.push_screen(Confirm(question, what.capitalize()), answered)

    @work(thread=True)
    def _control(self, bot: D.Bot, what: str) -> None:
        self.busy.add(str(bot.config))
        try:
            msgs = []
            if what in ("stop", "restart"):
                ok, msg = daemon.stop(bot.cfg)
                msgs.append(f"{bot.label}: {msg}")
            if what in ("start", "restart"):
                ok, msg = daemon.start(bot.cfg)
                msgs.append(f"{bot.label}: {msg}")
            self.call_from_thread(self.notify, "\n".join(msgs), severity="information" if ok else "error",
                                  timeout=8)
        finally:
            self.busy.discard(str(bot.config))
            self.call_from_thread(self.refresh_all)


def json_key(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True, default=str)


def run_monitor(config: str = "config.yaml") -> int:
    base = Path(config).resolve().parent
    if not D.discover(base):
        print(f"No bot is set up in {base} yet: starting the setup.")
        from .setup import run_setup
        return run_setup(config)
    Monitor(base).run()
    return 0
