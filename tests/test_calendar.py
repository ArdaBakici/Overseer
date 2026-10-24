"""The assistant's Google Calendar: reading events, free time, the agent's context, changes behind ✅ cards or with
↩️ Undo, reminders before events, and the login - against a fake Google."""

import json
import shutil
import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace as NS
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from overseer.assistant import calendar as C
from overseer.assistant.domain import AssistantDomain
from overseer.core.agent import Agent
from overseer.core.config import ConfigError
from overseer.core.ctl import ControlServer
from overseer.core.projects import Projects
from overseer.core.store import Store
from test_assistant import FakeNotifier, assistant_cfg


def at(day: date, hm: str) -> datetime:
    h, m = map(int, hm.split(":"))
    return C.local(datetime.combine(day, datetime.min.time()).replace(hour=h, minute=m))


class FakeGoogle:
    """Google's token endpoint and Calendar API, in memory."""

    def __init__(self, today: date):
        self.today = today
        self.calls = []
        self.refreshes = 0
        self.cals = [
            {"id": "me@gmail.com", "summary": "me@gmail.com", "summaryOverride": "Personal", "primary": True,
             "accessRole": "owner"},
            {"id": "work@group", "summary": "Work", "accessRole": "owner"},
            {"id": "holidays@group", "summary": "Holidays", "accessRole": "reader"},
            {"id": "old@group", "summary": "Old", "accessRole": "owner", "hidden": True},
        ]
        t, tm = today, today + timedelta(days=1)
        self.events = {
            "me@gmail.com": {
                "e1": self.ev("e1", "Gym", at(t, "07:00"), at(t, "08:00")),
                "e2": self.ev("e2", "Lunch with Sara", at(t, "12:00"), at(t, "13:00"), location="Cafe",
                              attendees=[{"email": "me@gmail.com", "self": True}, {"email": "sara@x.com"}]),
                "e3": self.ev("e3", "Dentist", at(tm, "10:00"), at(tm, "10:30")),
            },
            "work@group": {
                "w1": self.ev("w1", "Standup", at(t, "09:00"), at(t, "09:30")),
                "w2": self.ev("w2", "Skipped talk", at(t, "15:00"), at(t, "16:00"),
                              attendees=[{"email": "me@gmail.com", "self": True, "responseStatus": "declined"}]),
            },
            "holidays@group": {"h1": {"id": "h1", "summary": "Holiday", "start": {"date": tm.isoformat()},
                                      "end": {"date": (tm + timedelta(days=1)).isoformat()}}},
        }
        self.n = 0

    @staticmethod
    def ev(eid, title, start, end, **kw):
        return {"id": eid, "summary": title, "start": {"dateTime": start.isoformat()},
                "end": {"dateTime": end.isoformat()}, "status": "confirmed", **kw}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append((request.method, url.path, dict(url.params)))
        if url.host == "oauth2.googleapis.com":
            form = parse_qs(request.content.decode())
            if form["grant_type"] == ["refresh_token"]:
                if form["refresh_token"] == ["revoked"]:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                self.refreshes += 1
                return httpx.Response(200, json={"access_token": f"at{self.refreshes}", "expires_in": 3600})
            return httpx.Response(200, json={"access_token": "at0", "refresh_token": "rt", "expires_in": 3600,
                                             "scope": f"{C.LIST_SCOPE} {C.EVENTS_SCOPE}"})
        assert request.headers["authorization"].startswith("Bearer at")
        path = url.path.removeprefix("/calendar/v3")
        if path == "/users/me/calendarList":
            return httpx.Response(200, json={"items": self.cals})
        parts = path.split("/")  # /calendars/<cal>/events[/<id>]
        cal = parts[2]
        evs = self.events.setdefault(cal, {})
        if len(parts) == 4:
            if request.method == "POST":
                self.n += 1
                body = json.loads(request.content)
                eid = f"new{self.n}"
                evs[eid] = {**body, "id": eid, "status": "confirmed"}
                return httpx.Response(200, json=evs[eid])
            lo, hi = url.params["timeMin"], url.params["timeMax"]
            items = [e for e in evs.values() if self._end(e) > datetime.fromisoformat(lo)
                     and self._start(e) < datetime.fromisoformat(hi)]
            return httpx.Response(200, json={"items": sorted(items, key=self._start)})
        eid = parts[4]
        if eid not in evs:
            return httpx.Response(404, json={"error": {"message": "Not Found"}})
        if request.method == "GET":
            return httpx.Response(200, json=evs[eid])
        if request.method == "PATCH":
            evs[eid].update(json.loads(request.content))
            return httpx.Response(200, json=evs[eid])
        if request.method == "DELETE":
            del evs[eid]
            return httpx.Response(204)
        return httpx.Response(405)

    @staticmethod
    def _start(e):
        s = e["start"]
        return C.local(datetime.fromisoformat(s["dateTime"])) if "dateTime" in s else \
            C.local(datetime.fromisoformat(s["date"]))

    @staticmethod
    def _end(e):
        s = e["end"]
        return C.local(datetime.fromisoformat(s["dateTime"])) if "dateTime" in s else \
            C.local(datetime.fromisoformat(s["date"]))


@pytest.fixture
def cenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    cfg = assistant_cfg(tmp_path)
    cfg.calendar.enabled = True
    cfg.calendar.ask = ["primary"]
    cfg.calendar.auto = ["Work"]
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.local.db_path)
    notifier = FakeNotifier()
    domain = AssistantDomain(cfg, store, notifier)
    domain.prepare()
    agent = Agent(cfg, store, notifier, domain)
    ctl = ControlServer(cfg, agent, Projects(cfg, store, agent), notifier)
    google = FakeGoogle(date.today())
    domain.calendar.google.transport = httpx.MockTransport(google.handler)
    domain.calendar.google.save({"refresh_token": "rt", "scope": f"{C.LIST_SCOPE} {C.EVENTS_SCOPE}"})
    yield NS(cfg=cfg, store=store, notifier=notifier, domain=domain, agent=agent, ctl=ctl, google=google,
             cal=domain.calendar)
    shutil.rmtree(cfg.local.tmp(), ignore_errors=True)


def test_calendar_config(tmp_path):
    cfg = assistant_cfg(tmp_path)
    assert not cfg.calendar.enabled and cfg.calendar.ask == ["primary"]
    assert cfg.google_token_path == tmp_path / "app/state/assistant/google-token.json"
    assert cfg.google_token_path.parent in [p for p in cfg.local.secrets()]  # (agents can't read it)
    cfg.calendar.auto = "Work"
    with pytest.raises(ConfigError, match="calendar.auto"):
        cfg.validate()


async def test_reading_the_calendar(cenv):
    cal = cenv.cal
    await cal.sync()
    assert not cal.error and cenv.google.refreshes == 1
    assert [(c.name, c.mode) for c in cal.cals] == [("Personal", "ask"), ("Holidays", "read"), ("Work", "auto")]
    today = date.today()
    titles = [e.title for e in cal.cached_day(today)]
    assert titles == ["Gym", "Standup", "Lunch with Sara", "Skipped talk"]
    ctx = cal.context()
    assert "Today (4 events, 2h30 busy):" in ctx and "Tomorrow (2 events, 30m busy):" in ctx
    assert "Lunch with Sara @ Cafe [Personal, with others]" in ctx and "[Work, declined]" in ctx
    assert "tomorrow (all day) Holiday [Holidays]" in ctx
    # Free time: between the events that make you busy (not the declined one).
    events = cal.cached_day(today)
    slots = cal.free_slots(events, at(today, "06:00"), at(today, "18:00"), 30 * 60)
    assert [(a.strftime("%H:%M"), b.strftime("%H:%M")) for a, b in slots] == [
        ("06:00", "07:00"), ("08:00", "09:00"), ("09:30", "12:00"), ("13:00", "18:00")]
    # Only some calendars: read: [...] (ask and auto ones are always read).
    cenv.cfg.calendar.read = ["Holidays"]
    await cal.sync()
    assert [c.name for c in cal.cals] == ["Personal", "Holidays", "Work"]
    cenv.cfg.calendar.ask, cenv.cfg.calendar.auto = [], []
    await cal.sync()
    assert [c.name for c in cal.cals] == ["Holidays"]


async def test_a_login_google_no_longer_accepts(cenv):
    cenv.cal.google.save({"refresh_token": "revoked"})
    await cenv.cal.sync()
    assert "no longer accepts the login" in cenv.cal.error and "overseer login google" in cenv.cal.error
    assert "not read yet" in cenv.cal.context()


async def test_the_agent_sees_and_changes_the_calendar(cenv):
    from overseer.assistant.telegram import AssistantBot
    await cenv.cal.sync()
    prompt = cenv.agent.system_prompt(cenv.agent.general_session().id)
    assert "## Their calendar (Google)" in prompt and "Personal (primary): changes after they tap ✅" in prompt
    assert "Work: changes at once, with ↩️ Undo" in prompt and "Holidays: read-only" in prompt
    assert "<calendar>" in await cenv.domain.now_context()
    await cenv.ctl.start()
    try:
        s = cenv.agent.general_session()
        token = cenv.ctl.open(s.id)

        async def ov(*args):
            ok, text = await cenv.ctl.handle(token, ["cal", *args])
            return text
        listed = await ov("list", "today")
        assert "Standup" in listed and "Holiday" not in listed
        gym = next(e for e in cenv.cal.events if e.title == "Gym")
        lunch = next(e for e in cenv.cal.events if e.title == "Lunch with Sara")
        assert (await ov("list", "tomorrow", "--search", "dentist")).count("\n") == 0
        free = await ov("free", "tomorrow", "--from", "09:00", "--to", "12:00", "--min", "1h")
        assert free.endswith("free: 09:00–10:00 (1h), 10:30–12:00 (1h30)"), free
        assert "Personal (primary): changes after they tap ✅" in await ov("calendars")

        # In an "ask" calendar: a card, nothing changes until the tap.
        text = await ov("add", "Write the intro", "tomorrow 14:00", "2h")
        assert text.startswith("⏳ asked them") and len(cenv.google.events["me@gmail.com"]) == 3
        card = cenv.notifier.sent[-1]
        assert card.text.startswith("📅 <b>Add?</b> Write the intro · tomorrow 14:00–16:00 in Personal")
        (ok_label, ok_data), (_, no_data) = card.buttons[0]
        assert ok_label == "✅ Add" and card.session_id == s.id
        edits = []

        async def edit(q, text, buttons=None):
            edits.append((text, buttons))
        bot = AssistantBot(NS(_edit=edit, notifier=cenv.notifier), cenv.domain)
        action, _, arg = ok_data.partition(":")
        assert await bot.on_button(None, None, 1, action, arg)
        added = next(e for e in cenv.google.events["me@gmail.com"].values() if e["summary"] == "Write the intro")
        assert added["start"]["dateTime"].startswith((date.today() + timedelta(days=1)).isoformat() + "T14:00")
        text, buttons = edits[-1]
        assert text.startswith("📅 Added: Write the intro") and buttons[0][0][0] == "↩️ Undo"
        assert await bot.on_button(None, None, 1, action, arg) and "already answered" in edits[-1][0]
        assert any("They approved" in n for n in cenv.agent.take_notes(s.id))
        # ↩️ Undo removes it again.
        undo = buttons[0][0][1].partition(":")
        await bot.on_button(None, None, 1, undo[0], undo[2])
        assert edits[-1][0].startswith("↩️ <s>Added") and added["id"] not in cenv.google.events["me@gmail.com"]
        assert "can't be undone" in await cenv.cal.undo(undo[2])
        # ✖ No: nothing happens; the agent hears it.
        await ov("delete", gym.sid)
        action, _, arg = cenv.notifier.sent[-1].buttons[0][1][1].partition(":")
        await bot.on_button(None, None, 1, action, arg)
        assert edits[-1][0].startswith("✖ <s>Delete") and "e1" in cenv.google.events["me@gmail.com"]
        assert any("said no" in n for n in cenv.agent.take_notes(s.id))

        # In an "auto" calendar: done at once, with ↩️ Undo.
        standup = next(e for e in cenv.cal.events if e.title == "Standup")
        text = await ov("move", standup.sid, "tomorrow 11:00")
        assert text.startswith("✓ changed in Work"), text
        w1 = cenv.google.events["work@group"]["w1"]
        assert w1["start"]["dateTime"].startswith((date.today() + timedelta(days=1)).isoformat() + "T11:00")
        assert w1["end"]["dateTime"][11:16] == "11:30"  # (its length kept)
        notice = cenv.notifier.sent[-1]
        assert notice.text.startswith("📅 Changed: Standup in Work: today 09:00–09:30 → tomorrow 11:00–11:30")
        undo = notice.buttons[0][0][1].partition(":")
        await bot.on_button(None, None, 1, undo[0], undo[2])
        assert w1["start"]["dateTime"].startswith(date.today().isoformat() + "T09:00")
        text = await ov("delete", standup.sid)
        assert "w1" not in cenv.google.events["work@group"]
        undo = cenv.notifier.sent[-1].buttons[0][0][1].partition(":")
        await bot.on_button(None, None, 1, undo[0], undo[2])  # (deleted: put back as a copy)
        assert [e["summary"] for e in cenv.google.events["work@group"].values()].count("Standup") == 1

        # Never: events with other people, read-only calendars, or a calendar that doesn't exist.
        assert "other people invited" in await ov("edit", lunch.sid, "--title", "Lunch!")
        assert "read-only" in await ov("add", "x", "tomorrow 9:00", "--calendar", "Holidays")
        assert "no calendar 'Nope'" in await ov("add", "x", "tomorrow 9:00", "--calendar", "Nope")
        assert "no event 'zzzzzz'" in await ov("delete", "zzzzzz")
        assert "a length is like" in await ov("add", "x", "tomorrow 9:00", "forever")
        assert "before it starts" in await ov("add", "x", "tomorrow 9:00", "8:00")
        text = await ov("add", "Trip", "tomorrow", "--all-day", "--days", "3")
        assert "Trip · tomorrow – " in cenv.notifier.sent[-1].text and "(all day)" in cenv.notifier.sent[-1].text
    finally:
        await cenv.ctl.stop()


async def test_reminders_before_events(cenv, monkeypatch):
    cenv.cfg.calendar.event_reminders = 15
    now = datetime.now()
    soon = C.local(now + timedelta(minutes=20)).replace(second=0, microsecond=0)
    cenv.google.events["me@gmail.com"]["s1"] = FakeGoogle.ev("s1", "Call", soon, soon + timedelta(minutes=30),
                                                            location="Zoom")
    await cenv.cal.sync()
    rem = cenv.domain.reminders
    mine = [r for r in rem.pending() if r["text"].startswith("📅 Call")]
    assert len(mine) == 1 and mine[0]["text"] == f"📅 Call at {soon:%H:%M} · Zoom"
    assert abs(mine[0]["at"] - (soon.timestamp() - 15 * 60)) < 1
    await cenv.cal.sync()
    assert len([r for r in rem.pending() if r["text"].startswith("📅 Call")]) == 1  # (once)
    del cenv.google.events["me@gmail.com"]["s1"]  # cancelled: its reminder goes
    await cenv.cal.sync()
    assert not [r for r in rem.pending() if r["text"].startswith("📅 Call")]


async def test_digest_and_dashboard_show_today(cenv):
    from overseer.assistant import dashboard, digest
    parts = await digest.sections(cenv.domain, time.time() - 86400, time.time())
    cal = next(p for p in parts if p.startswith("📅"))
    assert "4 events" in cal and "07:00–08:00 Gym" in cal and "<i>@ Cafe</i>" in cal
    data = await dashboard.today(cenv.domain)
    assert [e["title"] for e in data["events"]] == ["Gym", "Standup", "Lunch with Sara", "Skipped talk"]


def test_login(cenv, tmp_path):
    cfg = cenv.cfg
    cenv.cal.google.path.unlink()
    out, links = [], []

    def read(prompt):
        url = next(x for x in out if x.startswith("1.")).split("\n\n")[1].strip()
        links.append(url)
        q = parse_qs(urlparse(url).query)
        assert q["redirect_uri"] == [C.REDIRECT] and q["code_challenge_method"] == ["S256"]
        assert C.EVENTS_SCOPE in q["scope"][0] and q["access_type"] == ["offline"]
        return f"{C.REDIRECT}?state={q['state'][0]}&code=4/abc&scope=x"
    http = httpx.Client(transport=httpx.MockTransport(cenv.google.handler))
    import overseer.assistant.calendar as mod
    orig = mod.Google.__init__

    def init(self, cfg, transport=None):  # (the calendars listed after logging in come from the fake too)
        orig(self, cfg, transport or httpx.MockTransport(cenv.google.handler))
    mod.Google.__init__ = init
    try:
        assert C.login(cfg, read=read, out=out.append, http=http) == 0
    finally:
        mod.Google.__init__ = orig
    token = json.loads(cfg.google_token_path.read_text())
    assert token["refresh_token"] == "rt" and oct(cfg.google_token_path.stat().st_mode)[-3:] == "600"
    shown = "\n".join(out)
    assert "Personal (primary): ask" in shown and "Work: auto" in shown and "Holidays: read" in shown
    # An address from another attempt is refused.
    assert C.login(cfg, read=lambda p: f"{C.REDIRECT}?state=other&code=x", out=out.append, http=http) == 1
    assert "another login attempt" in out[-1]
