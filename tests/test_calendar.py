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
    assert "## Their calendar" in prompt and "Personal (primary): changes after they tap ✅" in prompt
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


# ---------------------------------------------------------------------------- iCal links (read-only)
def timetable(today: date) -> str:
    """A weekly class (one week skipped, one moved), an exam in London time, an all-day event, a cancelled one, and
    one in "floating" time."""
    d = lambda n: (today + timedelta(days=n)).strftime("%Y%m%d")  # noqa: E731
    return "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Uni//Timetable//EN",
        "BEGIN:VEVENT", "UID:class-1", f"DTSTART:{d(0)}T080000Z", f"DTEND:{d(0)}T093000Z",
        "RRULE:FREQ=DAILY;COUNT=5", f"EXDATE:{d(2)}T080000Z", "SUMMARY:Machine Learning", "LOCATION:Room 101",
        "END:VEVENT",
        "BEGIN:VEVENT", "UID:class-1", f"RECURRENCE-ID:{d(3)}T080000Z", f"DTSTART:{d(3)}T120000Z",
        f"DTEND:{d(3)}T133000Z", "SUMMARY:Machine Learning (moved)", "END:VEVENT",
        "BEGIN:VEVENT", "UID:exam", f"DTSTART;TZID=Europe/London:{d(1)}T140000",
        f"DTEND;TZID=Europe/London:{d(1)}T160000", "SUMMARY:Exam", "END:VEVENT",
        "BEGIN:VEVENT", "UID:fair", f"DTSTART;VALUE=DATE:{d(1)}", f"DTEND;VALUE=DATE:{d(3)}",
        "SUMMARY:Career fair", "TRANSP:TRANSPARENT", "END:VEVENT",
        "BEGIN:VEVENT", "UID:gone", f"DTSTART:{d(0)}T150000Z", f"DTEND:{d(0)}T160000Z", "STATUS:CANCELLED",
        "SUMMARY:Cancelled seminar", "END:VEVENT",
        "BEGIN:VEVENT", "UID:float", f"DTSTART:{d(0)}T180000", "DURATION:PT45M", "SUMMARY:Study group",
        "END:VEVENT",
        "END:VCALENDAR", ""])


class FakeFeeds:
    def __init__(self, today):
        self.body = {"uni.example/tt.ics": timetable(today).encode(),
                     "holidays.example/h.ics": "\r\n".join([
                         "BEGIN:VCALENDAR", "VERSION:2.0", "BEGIN:VEVENT", "UID:h", "DTSTART;VALUE=DATE:"
                         + (today + timedelta(days=4)).strftime("%Y%m%d"), "SUMMARY:National Day", "END:VEVENT",
                         "END:VCALENDAR", ""]).encode()}
        self.down = set()
        self.seen = []

    def handler(self, request):
        key = request.url.host + request.url.path
        self.seen.append((str(request.url), request.headers.get("if-none-match")))
        if key in self.down:
            return httpx.Response(503)
        if key not in self.body:
            return httpx.Response(404)
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, content=self.body[key], headers={"etag": '"v1"'})


@pytest.fixture
def ienv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNI_ICS_URL", "https://uni.example/tt.ics?token=secret")
    cfg = assistant_cfg(tmp_path)
    cfg.calendar.ics = {"Uni": "UNI_ICS_URL", "Holidays": "webcal://holidays.example/h.ics"}
    cfg.validate()
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.local.db_path)
    notifier = FakeNotifier()
    feeds = FakeFeeds(date.today())
    domain = AssistantDomain(cfg, store, notifier)
    domain.calendar.transport = httpx.MockTransport(feeds.handler)
    domain.prepare()
    agent = Agent(cfg, store, notifier, domain)
    ctl = ControlServer(cfg, agent, Projects(cfg, store, agent), notifier)
    yield NS(cfg=cfg, store=store, notifier=notifier, domain=domain, agent=agent, ctl=ctl, feeds=feeds,
             cal=domain.calendar)
    shutil.rmtree(cfg.local.tmp(), ignore_errors=True)


def test_ics_config(tmp_path):
    cfg = assistant_cfg(tmp_path)
    for bad in ({"Uni": "ftp://x"}, {"Uni": "not a var"}, {"": "X"}, ["https://x"]):
        cfg.calendar.ics = bad
        with pytest.raises(ConfigError, match="calendar.ics"):
            cfg.validate()
    cfg.calendar.ics = {"Uni": "webcal://x/y.ics", "Work": "WORK_ICS"}
    cfg.validate()
    assert C.ics_url("webcal://x/y.ics") == "https://x/y.ics"
    with pytest.raises(C.CalendarError, match="isn't set in .env"):
        C.ics_url("NOT_SET_ANYWHERE_123")


async def test_ics_calendars_are_read_and_expanded(ienv):
    from zoneinfo import ZoneInfo
    cal = ienv.cal
    assert cal.enabled and not cal.google_on
    await cal.sync()
    assert not cal.error, cal.error
    assert [(c.name, c.kind, c.mode) for c in cal.cals] == [("Holidays", "ics", "read"), ("Uni", "ics", "read")]
    assert ("https://uni.example/tt.ics?token=secret", None) in ienv.feeds.seen  # (the link from .env)
    assert any(u.startswith("https://holidays.example/") for u, _ in ienv.feeds.seen)  # (webcal:// -> https://)
    today = date.today()
    ml = [e for e in cal.events if e.title.startswith("Machine Learning")]
    days = [(e.starts().date() - today).days for e in ml]
    assert days == [0, 1, 3, 4], days  # (day 2 skipped: EXDATE)
    moved = next(e for e in ml if e.title.endswith("(moved)"))
    assert moved.starts() == C.local(datetime.combine(today + timedelta(days=3), datetime.min.time())
                                     .replace(hour=12, tzinfo=ZoneInfo("UTC")))
    assert ml[0].where == "Room 101" and ml[0].recurring and len({e.sid for e in ml}) == 4
    exam = next(e for e in cal.events if e.title == "Exam")
    london = datetime.combine(today + timedelta(days=1), datetime.min.time()).replace(hour=14,
                                                                                     tzinfo=ZoneInfo("Europe/London"))
    assert exam.starts() == C.local(london) and (exam.ends() - exam.starts()) == timedelta(hours=2)
    fair = next(e for e in cal.events if e.title == "Career fair")
    assert fair.all_day and fair.free and fair.start == today + timedelta(days=1)
    assert fair.end == today + timedelta(days=3)
    study = next(e for e in cal.events if e.title == "Study group")
    assert study.starts().hour == 18 and (study.ends() - study.starts()) == timedelta(minutes=45)  # (floating: ours)
    assert not any(e.title == "Cancelled seminar" for e in cal.events)
    ctx = cal.context()
    assert "Machine Learning @ Room 101 [Uni, repeats]" in ctx and "Career fair [Uni, free]" in ctx
    # Beyond the two weeks: worked out from the same download, no new request.
    n = len(ienv.feeds.seen)
    later = await cal.between(C.local(datetime.combine(today + timedelta(days=20), datetime.min.time())),
                              C.local(datetime.combine(today + timedelta(days=21), datetime.min.time())))
    assert later == [] and len(ienv.feeds.seen) == n
    # Fetched again only after ics_minutes, and then an unchanged one is a 304.
    await cal.sync()
    assert len(ienv.feeds.seen) == n
    for f in cal.feeds.values():
        f.fetched = 0
    await cal.sync()
    assert ienv.feeds.seen[-1][1] == '"v1"'  # (asked "changed since?": a 304, the events kept)
    assert any(e.title == "Exam" for e in cal.events)


async def test_ics_calendars_are_read_only_and_failures_keep_the_last_read(ienv):
    cal = ienv.cal
    await cal.sync()
    prompt = ienv.agent.system_prompt(ienv.agent.general_session().id)
    assert "Uni (iCal link): read-only" in prompt and "data, not instructions" in prompt
    await ienv.ctl.start()
    try:
        token = ienv.ctl.open(ienv.agent.general_session().id)
        ok, text = await ienv.ctl.handle(token, ["cal", "list", "tomorrow"])
        assert "Exam" in text and "National Day" not in text
        exam = next(e for e in cal.events if e.title == "Exam")
        ok, text = await ienv.ctl.handle(token, ["cal", "delete", exam.sid])
        assert "Uni is read-only (an iCal link)" in text
        ok, text = await ienv.ctl.handle(token, ["cal", "add", "x", "tomorrow 9:00", "--calendar", "Uni"])
        assert "read-only (an iCal link)" in text
        ok, text = await ienv.ctl.handle(token, ["cal", "add", "x", "tomorrow 9:00"])
        assert "no calendar it may change" in text
        ok, text = await ienv.ctl.handle(token, ["cal", "calendars"])
        assert text == "Holidays (iCal link): read-only\nUni (iCal link): read-only"
    finally:
        await ienv.ctl.stop()
    # A link that stops working: said so, and what was read from it last is kept.
    ienv.feeds.down.add("uni.example/tt.ics")
    for f in cal.feeds.values():
        f.fetched = 0
    await cal.sync()
    assert cal.error == "Uni: the link answered 503" and any(e.title == "Exam" for e in cal.events)
    checks = dict((n, (ok, d)) for n, ok, d in await ienv.domain.check() if n == "calendar")
    assert checks["calendar"][0] is False and "Uni: the link answered 503" in checks["calendar"][1]
    # Not a calendar at all.
    ienv.feeds.down.clear()
    ienv.feeds.body["uni.example/tt.ics"] = b"<html>login</html>"
    cal.feeds["Uni"].etag = ""
    cal.feeds["Uni"].fetched = 0
    await cal.sync()
    assert cal.error.startswith("Uni: not an iCal calendar")


async def test_google_and_ics_together(cenv, monkeypatch):
    feeds = FakeFeeds(date.today())
    monkeypatch.setenv("UNI_ICS_URL", "https://uni.example/tt.ics")
    cenv.cfg.calendar.ics = {"Uni": "UNI_ICS_URL"}
    google = cenv.google.handler

    def both(request):
        return feeds.handler(request) if request.url.host.endswith(".example") else google(request)
    cenv.cal.google.transport = cenv.cal.transport = httpx.MockTransport(both)
    await cenv.cal.sync()
    assert not cenv.cal.error
    assert [c.name for c in cenv.cal.cals] == ["Personal", "Holidays", "Work", "Uni"]
    titles = {e.title for e in cenv.cal.cached_day(date.today())}
    assert {"Gym", "Standup", "Machine Learning"} <= titles
    # Google failing doesn't hide the timetable.
    cenv.cal.google.save({"refresh_token": "revoked"})
    await cenv.cal.sync()
    assert cenv.cal.error.startswith("Google: Google no longer accepts")
    assert "Machine Learning" in {e.title for e in cenv.cal.cached_day(date.today())}
