"""`overseer tui`: every bot of this Overseer folder, live, in the terminal. Read-only: it shows what the bots are doing
and never touches their conversations, Telegram or the HPC. Its only actions start and stop the bots themselves
(in the background: they keep running after the TUI closes).

It's run from the keyboard: ? shows every key.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import Reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, RichLog, Static,
                             TabbedContent, TabPane)

from ..core import daemon
from ..core.live import Style, render_event
from . import data as D

REFRESH = 2.0  # seconds
TABS = ["overview", "domain", "agents", "wakeups", "usage", "log"]
KEYS = [  # the help screen (?)
    ("Moving around", [
        ("← →  or  1-6", "previous / next tab, or straight to one"),
        ("[  ]", "previous / next bot"),
        ("↑ ↓  j k", "move in the list, or scroll"),
        ("PgUp PgDn  g G", "a page up / down; the top / the end"),
        ("Tab  Shift+Tab", "the next / previous pane (the bots, the list, the transcript)"),
        ("Enter", "open: a job's (or task's) details; a conversation's transcript"),
        ("Esc", "back (closes what Enter opened, clears the filter)"),
    ]),
    ("Looking", [
        ("/", "filter this tab's rows (Enter: keep it, Esc: clear it)"),
        ("f", "follow the transcript or log as it grows, or stay where you are"),
    ]),
    ("The bot (in the background: it keeps running when you quit)", [
        ("s", "start it"), ("x", "stop it"), ("r", "restart it"),
    ]),
    ("", [("?", "this help"), ("q", "quit (the bots keep running)")]),
]
NAMES = {"claude_code": "Claude Code", "codex": "Codex"}
ORIGINS = {"user": "you", "wakeup": "wake-up", "checkin": "check-in", "compact": "compaction",
           "subagent": "subagent", "routine": "routine"}


class QuietScreen(Screen):
    """The main screen, sending the terminal only what changed. (Textual repaints the whole screen when the focus
    moves - twice per tab switch - though the widgets that gain and lose it repaint themselves: over ssh, that's
    ~50 KB a key.)"""
    focused: Reactive = Reactive(None, repaint=False)


def for_slow_links(app: App) -> None:
    """Over ssh, no animations (each frame is sent); TEXTUAL_ANIMATIONS still decides if it's set."""
    if (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")) and "TEXTUAL_ANIMATIONS" not in os.environ:
        app.animation_level = "none"


class SteadyTabs(TabbedContent):
    """Tabs that only change when you change them. (Textual shows the tab of whatever gets the focus: as a tab is
    left, the focus goes to the next widget of the pane being hidden, or to a table being redrawn - and its tab came
    back.)"""

    def _on_tab_pane_focused(self, event: TabPane.Focused) -> None:
        event.stop()
        event.prevent_default()


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


class Help(ModalScreen[None]):
    DEFAULT_CSS = """
    Help { align: center middle; }
    #help { width: 84; height: auto; max-height: 90%; border: thick $primary; background: $surface; padding: 1 2; }
    """
    BINDINGS = [Binding("escape,q,question_mark,f1", "dismiss", "close")]

    def __init__(self, keys: list = KEYS):
        super().__init__()
        self.keys = keys

    def compose(self) -> ComposeResult:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", no_wrap=True)
        t.add_column()
        for title, rows in self.keys:
            t.add_row(Text(title, style="bold") if title else "", "")
            for key, what in rows:
                t.add_row(key, what)
            t.add_row("", "")
        with VerticalScroll(id="help"):
            yield Static(t)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class Details(ModalScreen[None]):
    """A row's whole text (a job's card, a task) - Enter on it; Esc closes."""
    DEFAULT_CSS = """
    Details { align: center middle; }
    #details-box { width: 90%; height: auto; max-height: 85%; border: thick $primary; background: $surface;
                   padding: 1 2; }
    """
    BINDINGS = [Binding("escape,q,enter", "dismiss", "close")]

    def __init__(self, title: str, text: str):
        super().__init__()
        self.title_text, self.text = title, text

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="details-box"):
            yield Static(Text(self.title_text, style="bold"))
            yield Static(Text(self.text or "(nothing more to show)"))

    def action_dismiss(self) -> None:
        self.dismiss(None)


class Monitor(App):
    TITLE = "Overseer"
    CSS = """
    #bots { width: 30; border-right: solid $primary-background; }
    #bots ListItem { padding: 0 1; }
    #summary { height: auto; padding: 0 1; background: $boost; }
    #tabs, #tabs > ContentSwitcher, #tabs TabPane { height: 1fr; }
    .table-title { padding: 1 1 0 1; text-style: bold; }
    #domain DataTable { height: auto; max-height: 20; }
    #details { height: auto; max-height: 14; padding: 0 1; border-top: solid $primary-background; }
    #conversations { width: 46%; }
    #transcript { border-left: solid $primary-background; }
    #overview, #usage { padding: 0 1; }
    #filter { dock: bottom; display: none; }
    #filter.shown { display: block; }
    *:focus { background-tint: $primary 8%; }
    """
    BINDINGS = [
        Binding("left", "move_tab(-1)", "tab", key_display="←→", priority=True),
        Binding("right", "move_tab(1)", show=False, priority=True),
        Binding("left_square_bracket", "move_bot(-1)", "bot", key_display="[ ]"),
        Binding("right_square_bracket", "move_bot(1)", show=False),
        *[Binding(str(i + 1), f"tab('{t}')", show=False) for i, t in enumerate(TABS)],
        Binding("j", "move(1)", show=False), Binding("k", "move(-1)", show=False),
        Binding("g", "ends(-1)", show=False), Binding("G", "ends(1)", show=False),
        Binding("slash", "filter", "filter", key_display="/"),
        Binding("f", "follow", "follow"),
        Binding("escape", "back", show=False),
        Binding("s", "start", "start"), Binding("x", "stop", "stop"), Binding("r", "restart", "restart"),
        Binding("question_mark,f1", "help", "help", key_display="?"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, base: Path):
        super().__init__()
        for_slow_links(self)
        self.base = base
        self.bots: list[D.Bot] = D.discover(base)
        self.selected = 0
        self.busy: set[str] = set()  # bots being started or stopped
        self.conversation: str | None = None
        self._shown: dict[str, object] = {}  # what each pane last showed (redrawn only when it changed)
        self.filters: dict[str, str] = {}  # tab -> its filter (/)
        self.follow = True  # the transcript and log scroll to new lines as they come (f)
        self._lines: dict[str, list[str]] = {}  # what a transcript / log shows (new lines are added, not redrawn)
        self._details: dict[str, list[str]] = {}
        self._rows: dict[str, list[list[str]]] = {}

    def get_default_screen(self) -> Screen:
        return QuietScreen(id="_default")

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(*[ListItem(Label(self._bot_line(b)), id=f"bot{i}") for i, b in enumerate(self.bots)],
                           id="bots")
            with Vertical():
                yield Static(id="summary")
                with SteadyTabs(id="tabs", initial="overview"):
                    with TabPane("Overview", id="overview"):
                        yield VerticalScroll(Static(id="overview-body"))
                    with TabPane("Bot", id="domain"):
                        yield VerticalScroll(id="domain-body")
                        yield Static(id="details")
                    with TabPane("Agents", id="agents"):
                        with Horizontal():
                            yield DataTable(id="conversations", cursor_type="row", zebra_stripes=True)
                            yield RichLog(id="transcript", wrap=True, markup=False, highlight=False, max_lines=5000)
                    with TabPane("Wake-ups", id="wakeups"):
                        yield DataTable(id="wakeup-table", cursor_type="row", zebra_stripes=True)
                    with TabPane("Usage", id="usage"):
                        yield VerticalScroll(Static(id="usage-body"))
                    with TabPane("Log", id="log"):
                        yield RichLog(id="log-body", wrap=False, markup=False, highlight=False, max_lines=5000)
        yield Input(placeholder="filter: type to narrow this tab's rows · Enter keep · Esc clear", id="filter")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#conversations", DataTable).add_columns("", "conversation", "project", "agent", "last")
        self.query_one("#wakeup-table", DataTable).add_columns("#", "conversation", "what", "when", "repeats")
        self.sub_title = f"{self.base}  ·  ? keys"
        self.refresh_all()
        self.set_interval(REFRESH, self.refresh_all)
        self.call_after_refresh(self._focus_tab)

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

    def _set(self, wid: str, renderable) -> None:
        """Show it, if it's not what's shown already (every repaint goes to the terminal: over ssh too)."""
        out = Console(file=io.StringIO(), width=200, color_system="truecolor", force_terminal=True)
        out.print(renderable)
        if self._changed(wid, out.file.getvalue()):
            self.query_one(wid, Static).update(renderable)

    # ------------------------------------------------------------------ refresh
    def refresh_all(self) -> None:
        for i, b in enumerate(self.bots):
            line = self._bot_line(b)
            if self._changed(f"bot{i}", line):
                self.query_one(f"#bot{i}", ListItem).query_one(Label).update(line)
        bot = self.bot
        if not bot:
            self._set("#summary", "No bot is set up in this folder yet: run [b]uv run overseer setup[/b].")
            return
        h = D.health(bot)
        self._set("#summary", self._summary(bot, h))
        con = D.connect_ro(bot.cfg.local.db_path) if bot.cfg else None
        try:
            active = self.query_one("#tabs", TabbedContent).active
            if active == "overview":
                self._set("#overview-body", self._overview(bot, h))
            if active == "domain":
                self._domain(h)
            if active == "agents":
                self._agents(bot, h, con)
            if active == "wakeups":
                self._wakeups(con)
            if active == "usage":
                self._set("#usage-body", self._usage(con))
            if active == "log":
                self._log(bot)
            self._domain_label(h)
        finally:
            if con:
                con.close()

    @on(TabbedContent.TabActivated)
    def tab_changed(self) -> None:
        self._forget()
        self._hide_filter()
        self.refresh_all()
        self.call_after_refresh(self._focus_tab)

    @on(ListView.Highlighted, "#bots")
    def bot_changed(self, event: ListView.Highlighted) -> None:
        if event.item is not None and event.item.id and int(event.item.id[3:]) != self.selected:
            self.selected = int(event.item.id[3:])
            self.conversation = None
            self._forget()
            self.refresh_all()

    def _forget(self) -> None:
        """Redraw the panes from scratch next time (another tab or bot)."""
        self._shown = {k: v for k, v in self._shown.items() if k.startswith("bot") or k == "#summary"}
        self._lines.clear()

    @property
    def tab(self) -> str:
        return self.query_one("#tabs", TabbedContent).active

    def _focus_tab(self) -> None:
        """Focus the tab's list (or text), so the arrow keys work in it at once."""
        target = {"overview": "#overview VerticalScroll", "domain": "#dt0", "agents": "#conversations",
                  "wakeups": "#wakeup-table", "usage": "#usage VerticalScroll", "log": "#log-body"}[self.tab]
        found = self.query(target)
        (found.first() if found else self.query_one("#domain-body")).focus()

    def _modal(self) -> bool:
        return len(self.screen_stack) > 1

    # ------------------------------------------------------------------ keys
    def action_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    def action_move_tab(self, step: int) -> None:
        focused = self.focused
        if isinstance(focused, Input):  # (in the filter, the arrows move its cursor)
            (focused.action_cursor_left if step < 0 else focused.action_cursor_right)()
            return
        if self._modal():
            return
        self.action_tab(TABS[(TABS.index(self.tab) + step) % len(TABS)])

    def action_move_bot(self, step: int) -> None:
        if self.bots:
            self.query_one("#bots", ListView).index = (self.selected + step) % len(self.bots)

    def action_move(self, step: int) -> None:
        w = self.focused
        name = ("cursor_down" if step > 0 else "cursor_up") if hasattr(w, "action_cursor_down") else \
            ("scroll_down" if step > 0 else "scroll_up")
        getattr(w, f"action_{name}", lambda: None)()

    def action_ends(self, end: int) -> None:
        w = self.focused
        if isinstance(w, ListView):
            w.index = len(w) - 1 if end > 0 else 0
            return
        name = ("scroll_bottom" if end > 0 else "scroll_top") if isinstance(w, DataTable) else \
            ("scroll_end" if end > 0 else "scroll_home")
        getattr(w, f"action_{name}", lambda: None)()

    def action_help(self) -> None:
        if not self._modal():
            self.push_screen(Help())

    def action_follow(self) -> None:
        self.follow = not self.follow
        for log in self.query(RichLog):
            log.auto_scroll = self.follow
            if self.follow:
                log.scroll_end(animate=False)
        self.notify("Following new lines" if self.follow else "Staying where you are (f: follow again)",
                    timeout=3)

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.value = self.filters.get(self.tab, "")
        box.add_class("shown")
        box.focus()

    def _hide_filter(self) -> None:
        box = self.query_one("#filter", Input)
        if box.has_class("shown"):
            box.remove_class("shown")

    @on(Input.Changed, "#filter")
    def filter_changed(self, event: Input.Changed) -> None:
        if not self.query_one("#filter", Input).has_class("shown"):
            return
        self.filters[self.tab] = event.value.strip()
        self._forget()
        self.refresh_all()

    @on(Input.Submitted, "#filter")
    def filter_kept(self) -> None:
        self._hide_filter()
        self._focus_tab()

    def action_back(self) -> None:
        box = self.query_one("#filter", Input)
        if self.focused is box or self.filters.get(self.tab):  # clear the filter
            self.filters.pop(self.tab, None)
            self._hide_filter()
            self._forget()
            self.refresh_all()
            self._focus_tab()
        elif self.focused is self.query_one("#transcript", RichLog):
            self.query_one("#conversations", DataTable).focus()

    def _matches(self, *cells) -> bool:
        words = self.filters.get(self.tab, "").lower().split()
        blob = " ".join(str(c) for c in cells).lower()
        return all(w in blob for w in words)

    def _show_lines(self, wid: str, lines: list[Text]) -> None:
        """A transcript or log: new lines are added (you stay where you are unless following), not redrawn."""
        log = self.query_one(wid, RichLog)
        log.auto_scroll = self.follow
        plain = [t.plain for t in lines]
        old = self._lines.get(wid)
        # (the log is its last lines: when new ones come, the first go. What's shown carries on, if it does)
        kept = next((len(old) - k for k in range(len(old)) if old[k] == plain[0] and plain[:len(old) - k] == old[k:]),
                    None) if old and plain else None
        if old is not None and kept is not None:
            new = lines[kept:]
        elif old is not None and not old:
            new = lines
        else:
            y = log.scroll_y
            log.clear()
            new = lines
            if not self.follow and old is not None:  # (the same text, redrawn: stay where you were)
                self.call_after_refresh(lambda: log.scroll_to(y=y, animate=False))
        for t in new:
            log.write(t, scroll_end=self.follow)
        self._lines[wid] = plain

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
                parts.append("not restarted after a crash (local.restart_on_crash: true turns it on)")
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
        if self._changed("domain", json_key([tables, self.filters.get("domain", "")])):
            self._build_domain(tables)

    @work(exclusive=True, group="domain")
    async def _build_domain(self, tables: list[dict]) -> None:
        body = self.query_one("#domain-body", VerticalScroll)
        # (where each table's cursor was, by its row: it stays on that row)
        was = {dt.id: (self._rows.get(dt.id) or [[]])[dt.cursor_row][:2] if dt.row_count else None
               for dt in body.query(DataTable)}
        focused = self.focused.id if isinstance(self.focused, DataTable) else None
        await body.remove_children()  # (before the new tables: they reuse the ids)
        self._details, self._rows = {}, {}
        if not tables:
            await body.mount(Static("Nothing yet: the bot shows its jobs (or its day) here once it has run.",
                                    classes="table-title"))
        for i, tb in enumerate(tables):
            tid = f"dt{i}"
            pairs = [(row, d) for row, d in zip(tb["rows"], (tb.get("details") or []) + [""] * len(tb["rows"]))
                     if self._matches(*row, d)]
            dt = DataTable(id=tid, cursor_type="row", zebra_stripes=True)
            shown = f"{len(pairs)} of {len(tb['rows'])}" if len(pairs) != len(tb["rows"]) else str(len(tb["rows"]))
            await body.mount(Label(f"{tb['title']} ({shown})", classes="table-title"), dt)
            dt.add_columns(*tb["columns"])
            rows = [[str(c) for c in row] for row, _ in pairs]
            for row in rows:
                dt.add_row(*row)
            self._rows[tid], self._details[tid] = rows, [d for _, d in pairs]
            key = was.get(tid)
            at = next((n for n, row in enumerate(rows) if key and row[:2] == key), 0)
            if at:
                dt.move_cursor(row=at, animate=False)
            if self.tab == "domain" and (  # (only if you're still there: focusing it would bring its tab back)
                    tid == focused or (focused is None and i == 0 and not isinstance(self.focused, (Input, ListView)))):
                dt.focus()

    @on(DataTable.RowSelected)
    def opened(self, event: DataTable.RowSelected) -> None:
        """Enter on a row: a job's (or task's) whole text; a conversation's transcript."""
        tid = event.data_table.id or ""
        if tid.startswith("dt"):
            rows, details = self._rows.get(tid) or [], self._details.get(tid) or []
            if event.cursor_row < len(rows):
                self.push_screen(Details(" · ".join(c for c in rows[event.cursor_row] if c),
                                         details[event.cursor_row] if event.cursor_row < len(details) else ""))
        elif tid == "conversations":
            self.query_one("#transcript", RichLog).focus()

    @on(DataTable.RowHighlighted)
    def row(self, event: DataTable.RowHighlighted) -> None:
        tid = event.data_table.id or ""
        if tid.startswith("dt"):
            details = self._details.get(tid) or []
            text = details[event.cursor_row] if event.cursor_row < len(details) else ""
            self.query_one("#details", Static).update(Text(text))
        elif tid == "conversations":
            key = event.row_key.value if event.row_key else None
            if key and key != self.conversation:
                self.conversation = key
                self._shown.pop("transcript", None)
                self._lines.pop("#transcript", None)
                self._transcript(self.bot)

    def _agents(self, bot: D.Bot, h: D.Health, con) -> None:
        running = {r["session"] for r in h.status.get("runs") or []} if h.running else set()
        convs = [c for c in D.conversations(con)
                 if self._matches(c["name"], c["id"], c["project"], c["backend"], c["role"])]
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
            ids = [c["id"] for c in convs]
            if ids and self.conversation not in ids:
                self.conversation = ids[0]
            if self.conversation in ids:  # (the cursor stays on the conversation you were looking at)
                dt.move_cursor(row=ids.index(self.conversation), animate=False)
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
        if stamp is None:
            self._show_lines("#transcript", [
                Text("No live log for this conversation (it starts with its next request).", style="dim")])
            return
        st, last_message, lines = Style(True), None, [Text(f"── {self.conversation}", style="dim")]
        for ev in D.transcript_events(bot.state_dir, self.conversation):
            lines += [Text.from_ansi(line) for line in render_event(ev, st, last_message=last_message)]
            if ev.get("kind") in ("message", "turn"):
                last_message = ev.get("text") if ev["kind"] == "message" else None
        self._show_lines("#transcript", lines)

    def _wakeups(self, con) -> None:
        rows = []
        for w in D.wakeups(con):
            what = {"jobs": f"job(s) {w['jobs']} end", "timer": "timer"}.get(w["kind"], f"{w['kind']} {w['jobs']}".strip())
            if w["note"]:
                what += f": {w['note']}"
            every = f"every {D.ago(time.time() - w['every'])}" if w["every"] else ""
            row = [str(w["id"]), w["session"] or "", what[:90], D.until(w["due"]) if w["due"] else "when it ends", every]
            if self._matches(*row):
                rows.append(row)
        if not self._changed("wakeups", rows):
            return
        dt = self.query_one("#wakeup-table", DataTable)
        was = dt.coordinate_to_cell_key(dt.cursor_coordinate).row_key.value if dt.row_count else None
        dt.clear()
        for row in rows:
            dt.add_row(*row, key=row[0])
        if was in {r[0] for r in rows}:  # (the cursor stays on its wake-up)
            dt.move_cursor(row=dt.get_row_index(was), animate=False)

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
        lines = [line for line in D.log_tail(bot.state_dir) if self._matches(line)]
        if not self._changed("log", (len(lines), lines[-1] if lines else "", self.filters.get("log", ""))):
            return
        if not lines:
            self._show_lines("#log-body", [Text("No log yet: it starts when the bot runs." if not self.filters.get("log")
                                                else "No line matches the filter (Esc clears it).", style="dim")])
            return
        self._show_lines("#log-body", [
            Text(line, style="red" if " ERROR " in line or "Traceback" in line else
                 "yellow" if " WARNING " in line else "") for line in lines])

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
