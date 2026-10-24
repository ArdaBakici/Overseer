"""Your calendars, for the assistant: Google Calendar, and read-only ones from iCal links (calendar.ics: a timetable,
a shared or published calendar). Overseer reads the next two weeks of events every few minutes (the agent sees today's
and tomorrow's in its context, and asks for more with `ov cal`), and makes the changes the agent asks for - in the
Google calendars you allow: after you tap ✅ on a card (calendar.ask), or on its own with ↩️ Undo (calendar.auto).

The agent never holds the Google login: the token is in the bot's state folder, which agents can't read, and only
Overseer talks to Google. It can't invite anyone, and events with other people are left alone (change those in
Google Calendar yourself). Log in once with `overseer login google -c <config>` (works on a headless machine).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from ..core.notify import Outgoing, esc

log = logging.getLogger(__name__)

API = "https://www.googleapis.com/calendar/v3"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
LIST_SCOPE = "https://www.googleapis.com/auth/calendar.calendarlist.readonly"
EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"
EVENTS_RO_SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"
REDIRECT = "http://127.0.0.1:8765/"  # (nothing listens there: you copy the address the browser ends up at)
WINDOW_DAYS = 14  # what's kept in memory, from today
KEY = "assistant:cal:"  # + id: a change waiting for ✅ / done (for ↩️ Undo)
REMINDERS_KEY = "assistant:cal_reminders"  # event -> the reminder set for it
KEEP_SECONDS = 7 * 86400  # how long a change can still be approved or undone
COPY_FIELDS = ("summary", "start", "end", "location", "description", "colorId", "transparency", "visibility",
               "reminders")


class CalendarError(Exception):
    pass


# ---------------------------------------------------------------------------- times
def local(dt: datetime) -> datetime:
    """In this bot's time zone (local.timezone), aware."""
    return dt.astimezone() if dt.tzinfo else dt.replace(tzinfo=datetime.now().astimezone().tzinfo)


def _parse_time(t: dict) -> tuple[datetime | date, bool]:
    if "date" in t:
        return date.fromisoformat(t["date"]), True
    return local(datetime.fromisoformat(t["dateTime"].replace("Z", "+00:00"))), False


def _google_time(v: datetime | date, tz: str) -> dict:
    if isinstance(v, datetime):
        return {"dateTime": local(v).isoformat(), **({"timeZone": tz} if tz else {})}
    return {"date": v.isoformat()}


def span_text(start, end, all_day: bool, today: date | None = None) -> str:
    """"Thu 30 Oct 14:00–16:00", "today 14:00–16:00", "Sat 1 Nov (all day)", "Sat 1 – Mon 3 Nov (all day)"."""
    today = today or date.today()

    def day(d: date) -> str:
        return "today" if d == today else "tomorrow" if d == today + timedelta(days=1) else f"{d:%a} {d.day} {d:%b}"
    if all_day:
        last = end - timedelta(days=1)
        return f"{day(start)} (all day)" if last <= start else f"{day(start)} – {day(last)} (all day)"
    if start.date() == end.date():
        return f"{day(start.date())} {start:%H:%M}–{end:%H:%M}"
    return f"{day(start.date())} {start:%H:%M} – {day(end.date())} {end:%H:%M}"


def re_clock(text: str) -> bool:
    """"16:00", "4pm": a time of day (not a length)."""
    import re
    return bool(re.fullmatch(r"\d{1,2}:\d{2}|\d{1,2}(:\d{2})?\s*(am|pm)", text.strip().lower()))


def duration_text(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}" if m >= 60 and m % 60 else f"{m // 60}h" if m >= 60 else f"{m}m"


# ---------------------------------------------------------------------------- what Overseer knows
@dataclass
class Cal:
    id: str
    name: str
    primary: bool = False
    mode: str = "read"  # read | ask | auto
    role: str = "reader"  # Google's accessRole: owner | writer | reader | freeBusyReader
    tz: str = ""
    kind: str = "google"  # google | ics (an iCal link: read-only)


@dataclass
class Event:
    cal: str
    cal_name: str
    id: str
    title: str
    start: datetime | date
    end: datetime | date
    all_day: bool
    where: str = ""
    note: str = ""
    others: int = 0  # other people invited (then it's left alone)
    declined: bool = False
    free: bool = False  # "show as available"
    recurring: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def sid(self) -> str:
        return short_id(self.cal, self.id)

    def starts(self) -> datetime:
        return self.start if isinstance(self.start, datetime) else local(datetime.combine(self.start, dtime(0, 0)))

    def ends(self) -> datetime:
        return self.end if isinstance(self.end, datetime) else local(datetime.combine(self.end, dtime(0, 0)))

    def line(self, today: date | None = None, show_cal: bool = True) -> str:
        bits = [self.sid, span_text(self.start, self.end, self.all_day, today), self.title or "(no title)"]
        if self.where:
            bits.append(f"@ {self.where[:60]}")
        tags = [self.cal_name] if show_cal else []
        tags += [t for t, on in (("with others", self.others), ("declined", self.declined), ("free", self.free),
                                 ("repeats", self.recurring)) if on]
        return " ".join(bits) + (f" [{', '.join(tags)}]" if tags else "")

    def to_dict(self) -> dict:
        return {"id": self.sid, "calendar": self.cal_name, "title": self.title, "where": self.where,
                "all_day": self.all_day, "start": self.starts().timestamp(), "end": self.ends().timestamp(),
                "others": self.others, "declined": self.declined}


def short_id(cal: str, eid: str) -> str:
    return hashlib.sha1(f"{cal}/{eid}".encode()).hexdigest()[:6]


def make_event(cal: Cal, item: dict) -> Event | None:
    if item.get("status") == "cancelled" or "start" not in item:
        return None
    start, all_day = _parse_time(item["start"])
    end, _ = _parse_time(item.get("end") or item["start"])
    me = next((a for a in item.get("attendees") or [] if a.get("self")), None)
    others = [a for a in item.get("attendees") or [] if not a.get("self") and not a.get("resource")]
    return Event(cal=cal.id, cal_name=cal.name, id=item["id"], title=item.get("summary", ""), start=start, end=end,
                 all_day=all_day, where=item.get("location", ""), note=(item.get("description") or "")[:500],
                 others=len(others), declined=bool(me and me.get("responseStatus") == "declined"),
                 free=item.get("transparency") == "transparent", recurring=bool(item.get("recurringEventId")),
                 raw=item)


# ---------------------------------------------------------------------------- Google
class Google:
    """Google's Calendar API with your own OAuth client (a Desktop app); the token lives in the state folder."""

    def __init__(self, cfg, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.path: Path = cfg.google_token_path
        self.transport = transport
        self._token: dict | None = None
        self._lock = asyncio.Lock()

    @property
    def client(self) -> tuple[str, str]:
        c = self.cfg.calendar
        cid, secret = os.environ.get(c.client_id_env, ""), os.environ.get(c.client_secret_env, "")
        if not cid or not secret:
            raise CalendarError(f"{c.client_id_env} / {c.client_secret_env} aren't set in .env (your OAuth client's id "
                                "and secret)")
        return cid, secret

    def token(self) -> dict | None:
        if self._token is None:
            try:
                self._token = json.loads(self.path.read_text())
            except (OSError, ValueError):
                return None
        return self._token

    def logged_in(self) -> bool:
        return bool((self.token() or {}).get("refresh_token"))

    def save(self, token: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(token, fh)
        os.replace(tmp, self.path)
        self._token = token

    def writes(self) -> bool:
        return EVENTS_SCOPE in (self.token() or {}).get("scope", "").split()

    # ---- logging in
    def auth_url(self, verifier: str, state: str, write: bool) -> str:
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        return AUTH_URL + "?" + urlencode({
            "client_id": self.client[0], "redirect_uri": REDIRECT, "response_type": "code",
            "scope": " ".join([LIST_SCOPE, EVENTS_SCOPE if write else EVENTS_RO_SCOPE]),
            "access_type": "offline", "prompt": "consent", "state": state,
            "code_challenge": challenge, "code_challenge_method": "S256"})

    def exchange(self, code: str, verifier: str, http: httpx.Client | None = None) -> dict:
        cid, secret = self.client
        with http or httpx.Client(timeout=30) as h:
            r = h.post(TOKEN_URL, data={"code": code, "client_id": cid, "client_secret": secret,
                                        "redirect_uri": REDIRECT, "grant_type": "authorization_code",
                                        "code_verifier": verifier})
        data = _json(r)
        if r.status_code != 200 or "refresh_token" not in data:
            raise CalendarError(f"Google refused: {data.get('error_description') or data.get('error') or r.text[:200]}")
        token = {"refresh_token": data["refresh_token"], "access_token": data.get("access_token", ""),
                 "expires": time.time() + float(data.get("expires_in", 0)) - 60, "scope": data.get("scope", "")}
        self.save(token)
        return token

    # ---- calls
    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=30, transport=self.transport)

    async def _access(self, force: bool = False) -> str:
        async with self._lock:
            tok = self.token()
            if not tok or not tok.get("refresh_token"):
                raise CalendarError("not logged in to Google: overseer login google -c <config>")
            if not force and tok.get("access_token") and tok.get("expires", 0) > time.time():
                return tok["access_token"]
            cid, secret = self.client
            async with self._http() as h:
                r = await h.post(TOKEN_URL, data={"client_id": cid, "client_secret": secret,
                                                  "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"})
            data = _json(r)
            if r.status_code != 200:
                if data.get("error") == "invalid_grant":
                    raise CalendarError("Google no longer accepts the login (it was revoked or expired - in Google "
                                        "Cloud, an app left in \"Testing\" loses it after 7 days: publish it). Log in "
                                        "again: overseer login google -c <config>")
                raise CalendarError(f"Google: {data.get('error_description') or data.get('error') or r.status_code}")
            tok = {**tok, "access_token": data["access_token"],
                   "expires": time.time() + float(data.get("expires_in", 3600)) - 60}
            if data.get("scope"):
                tok["scope"] = data["scope"]
            self.save(tok)
            return tok["access_token"]

    async def call(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> dict:
        for attempt in (0, 1):
            token = await self._access(force=attempt == 1)
            try:
                async with self._http() as h:
                    r = await h.request(method, API + path, params=params, json=body,
                                        headers={"Authorization": f"Bearer {token}"})
            except httpx.HTTPError as e:
                raise CalendarError(f"couldn't reach Google ({type(e).__name__})") from None
            if r.status_code == 401 and attempt == 0:
                continue
            if r.status_code in (200, 201, 204):
                return _json(r) if r.content else {}
            err = _json(r).get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            raise CalendarError(f"Google: {msg or r.status_code}")
        raise CalendarError("Google: not authorised")

    async def calendars(self) -> list[dict]:
        return await self._pages("/users/me/calendarList", {"minAccessRole": "freeBusyReader"})

    async def events(self, cal: str, start: datetime, end: datetime) -> list[dict]:
        return await self._pages(f"/calendars/{_q(cal)}/events", {
            "timeMin": start.isoformat(), "timeMax": end.isoformat(), "singleEvents": "true",
            "orderBy": "startTime", "maxResults": "250"})

    async def _pages(self, path: str, params: dict) -> list[dict]:
        out, token = [], None
        for _ in range(20):
            data = await self.call("GET", path, {**params, **({"pageToken": token} if token else {})})
            out += data.get("items", [])
            token = data.get("nextPageToken")
            if not token:
                break
        return out


def _q(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


def _json(r: httpx.Response) -> dict:
    try:
        v = r.json()
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def login(cfg, read=input, out=print, http: httpx.Client | None = None) -> int:
    """`overseer login google`: the consent link, then the address the browser ended up at."""
    g = Google(cfg)
    c = cfg.calendar
    try:
        g.client
    except CalendarError as e:
        out(f"{e}.\nCreate one: console.cloud.google.com → APIs & Services → enable the Google Calendar API; "
            "OAuth consent screen (External, add yourself as a test user, then publish it so the login doesn't expire "
            "after 7 days); Credentials → Create credentials → OAuth client ID → Desktop app. Put its id and secret in "
            f".env as {c.client_id_env}=... and {c.client_secret_env}=...")
        return 1
    verifier, state = secrets.token_urlsafe(48), secrets.token_urlsafe(12)
    write = bool(c.ask or c.auto)
    out("1. Open this link (on any device), sign in and allow access"
        + (" (see your calendars, and change events)" if write else " (read-only: calendar.ask and auto are empty)")
        + f":\n\n{g.auth_url(verifier, state, write)}\n")
    out("2. The browser then fails to open a page on 127.0.0.1 - that's expected. Copy that page's whole address "
        "(it has ?code=...) and paste it here.\n")
    pasted = read("Address: ").strip()
    q = parse_qs(urlparse(pasted).query) if "code=" in pasted else {"code": [pasted], "state": [state]}
    if q.get("error"):
        out(f"Google said: {q['error'][0]}")
        return 1
    if q.get("state", [""])[0] != state:
        out("That address is from another login attempt: run this again and use the new link.")
        return 1
    try:
        g.exchange(q["code"][0], verifier, http)
    except (CalendarError, httpx.HTTPError) as e:
        out(f"Login failed: {e}")
        return 1
    out(f"✓ Logged in; the token is in {g.path} (only Overseer reads it).")
    try:
        cals = asyncio.run(Calendar.fetch_calendars(cfg, g))
    except CalendarError as e:
        out(f"...but listing the calendars failed: {e}")
        return 1
    out("\nYour calendars (what the assistant may do, from calendar.read / ask / auto in the config):")
    for cal in cals:
        out(f"  {cal.name}{' (primary)' if cal.primary else ''}: {cal.mode}  · id {cal.id}")
    if not c.enabled:
        out("\nSet calendar.enabled: true in the config, then restart the bot.")
    return 0


# ---------------------------------------------------------------------------- iCal links (read-only)
ICS_MAX_BYTES = 10 * 1024 * 1024


def ics_url(source: str) -> str:
    """calendar.ics's value: the link itself (webcal:// works too), or the name of a variable in .env holding it."""
    s = source.strip()
    if "://" not in s:
        s = os.environ.get(s, "").strip()
        if not s:
            raise CalendarError(f"{source} isn't set in .env")
    if s.lower().startswith("webcal://"):
        s = "https://" + s[9:]
    if not s.lower().startswith(("https://", "http://")):
        raise CalendarError("not an http(s) or webcal link")
    return s


class Feed:
    """One iCal link: fetched again at most every few minutes (unchanged ones cost a 304), expanded locally -
    repeating events, their exceptions and time zones by recurring-ical-events."""

    def __init__(self, name: str, source: str, transport: httpx.AsyncBaseTransport | None = None):
        self.name = name
        self.source = source
        self.transport = transport
        self.ical = None
        self.series: set[str] = set()  # the UIDs of repeating events
        self.etag = self.modified = ""
        self.fetched = 0.0

    def cal(self) -> Cal:
        return Cal(id=f"ics:{self.name}", name=self.name, mode="read", kind="ics")

    async def refresh(self, every: float) -> None:
        if self.ical is not None and time.time() - self.fetched < every:
            return
        url = ics_url(self.source)
        headers = {"User-Agent": "Overseer (calendar)"}
        if self.ical is not None:  # (unchanged since: a 304, nothing to read again)
            headers |= {**({"If-None-Match": self.etag} if self.etag else {}),
                        **({"If-Modified-Since": self.modified} if self.modified else {})}
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True, transport=self.transport) as h:
                async with h.stream("GET", url, headers=headers) as r:
                    if r.status_code == 304 and self.ical is not None:
                        self.fetched = time.time()
                        return
                    if r.status_code != 200:
                        raise CalendarError(f"the link answered {r.status_code}")
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > ICS_MAX_BYTES:
                            raise CalendarError(f"bigger than {ICS_MAX_BYTES // 2**20} MB")
                    etag, modified = r.headers.get("etag", ""), r.headers.get("last-modified", "")
        except httpx.HTTPError as e:
            raise CalendarError(f"couldn't fetch it ({type(e).__name__})") from None
        self.ical = await asyncio.to_thread(parse_ics, body)
        self.series = {str(c.get("UID", "")) for c in self.ical.walk("VEVENT")
                       if any(k in c for k in ("RRULE", "RDATE", "RECURRENCE-ID"))}
        self.etag, self.modified, self.fetched = etag, modified, time.time()

    def events(self, cal: Cal, start: datetime, end: datetime) -> list[Event]:
        if self.ical is None:
            return []
        import recurring_ical_events
        try:
            items = recurring_ical_events.of(self.ical, skip_bad_series=True).between(start, end)
        except Exception as e:  # (a broken repeat rule in someone else's calendar: say so, don't crash)
            raise CalendarError(f"couldn't read its events ({type(e).__name__}: {str(e)[:100]})") from None
        return [ev for item in items if (ev := ics_event(cal, item, self.series))]


def parse_ics(body: bytes):
    import icalendar
    try:
        cal = icalendar.Calendar.from_ical(body)
    except Exception as e:
        raise CalendarError(f"not an iCal calendar ({str(e)[:100]})") from None
    if not isinstance(cal, icalendar.Calendar):
        raise CalendarError("not an iCal calendar")
    return cal


def _ics_time(v) -> tuple[datetime | date, bool]:
    if isinstance(v, datetime):
        return local(v), False  # (no time zone = "floating": this bot's)
    return v, True


def ics_event(cal: Cal, item, series: set[str] = frozenset()) -> Event | None:
    if str(item.get("STATUS", "")).upper() == "CANCELLED" or "DTSTART" not in item:
        return None
    start, all_day = _ics_time(item.decoded("DTSTART"))
    if "DTEND" in item:
        end, _ = _ics_time(item.decoded("DTEND"))
    elif "DURATION" in item:
        end = start + item.decoded("DURATION")
    else:
        end = start + timedelta(days=1) if all_day else start
    if all_day and isinstance(end, datetime):
        end = end.date()
    uid = str(item.get("UID", "")) or hashlib.sha1(str(item.get("SUMMARY", "")).encode()).hexdigest()[:12]
    stamp = start.isoformat()
    attendees = item.get("ATTENDEE") or []
    people = len(attendees) if isinstance(attendees, list) else 1
    return Event(cal=cal.id, cal_name=cal.name, id=f"{uid}@{stamp}", title=str(item.get("SUMMARY", "")),
                 start=start, end=end, all_day=all_day, where=str(item.get("LOCATION", "")),
                 note=str(item.get("DESCRIPTION", ""))[:500], others=max(0, people - 1),
                 free=str(item.get("TRANSP", "")).upper() == "TRANSPARENT",
                 recurring=str(item.get("UID", "")) in series)


# ---------------------------------------------------------------------------- the service
class Calendar:
    """Owned by the AssistantDomain: the events cache, its sync loop, reminders before events, and changes."""

    def __init__(self, domain, transport: httpx.AsyncBaseTransport | None = None):
        self.domain = domain
        self.cfg = domain.cfg
        self.store = domain.store
        self.google = Google(domain.cfg, transport)
        self.transport = transport
        self.feeds: dict[str, Feed] = {}  # name -> an iCal link
        self._sources: dict[str, tuple[list[Cal], list[Event]]] = {}  # "google" / "ics:<name>" -> what was read last
        self.cals: list[Cal] = []
        self.events: list[Event] = []
        self.window: tuple[datetime, datetime] | None = None
        self.synced = 0.0
        self.error = ""
        self.ids: dict[str, tuple[str, str]] = {}  # short id -> (calendar, event id)

    @property
    def enabled(self) -> bool:
        return self.google_on or bool(self.cfg.calendar.ics)

    @property
    def google_on(self) -> bool:
        return bool(self.cfg.calendar.enabled)

    def feed(self, name: str) -> Feed:
        source = self.cfg.calendar.ics[name]
        f = self.feeds.get(name)
        if f is None or f.source != source:
            f = self.feeds[name] = Feed(name, source, self.transport)
        return f

    # ---- calendars and their modes
    @staticmethod
    def mode_for(c, item: dict) -> str | None:
        """read | ask | auto, or None if it isn't one to read."""
        names = {item.get("id", "").lower(), (item.get("summaryOverride") or item.get("summary") or "").lower()}
        if item.get("primary"):
            names.add("primary")

        def hit(lst) -> bool:
            return any(x.strip().lower() in names for x in lst)
        writable = item.get("accessRole") in ("owner", "writer")
        if writable and hit(c.auto):
            return "auto"
        if writable and hit(c.ask):
            return "ask"
        if not c.read or hit(c.read) or hit(c.ask) or hit(c.auto):
            return "read"
        return None

    @classmethod
    async def fetch_calendars(cls, cfg, google: Google) -> list[Cal]:
        out = []
        for item in await google.calendars():
            mode = cls.mode_for(cfg.calendar, item)
            if mode is None or item.get("hidden"):
                continue
            if mode != "read" and not google.writes():
                mode = "read"  # (logged in read-only)
            out.append(Cal(id=item["id"], name=item.get("summaryOverride") or item.get("summary") or item["id"],
                           primary=bool(item.get("primary")), mode=mode, role=item.get("accessRole", "reader"),
                           tz=item.get("timeZone", "")))
        out.sort(key=lambda c: (not c.primary, c.name.lower()))
        return out

    def cal(self, name: str) -> Cal:
        n = name.strip().lower()
        for c in self.cals:
            if n in (c.id.lower(), c.name.lower()) or (n == "primary" and c.primary):
                return c
        raise CalendarError(f"no calendar {name!r}: {', '.join(c.name for c in self.cals) or '(none yet)'}")

    def default_cal(self) -> Cal:
        writable = [c for c in self.cals if c.mode in ("ask", "auto")]
        if not writable:
            raise CalendarError("no calendar it may change: add one to calendar.ask or calendar.auto in the config")
        return next((c for c in writable if c.primary), writable[0])

    # ---- reading
    async def sync(self) -> None:
        """Read every source again; one that fails keeps what was read from it last (and says so)."""
        if not self.enabled:
            return
        start = local(datetime.combine(date.today(), dtime(0, 0)))
        end = start + timedelta(days=WINDOW_DAYS)
        sources: list[tuple[str, Any]] = []
        if self.google_on:
            sources.append(("google", self._read_google))
        sources += [(f"ics:{n}", lambda s, e, n=n: self._read_feed(n, s, e)) for n in self.cfg.calendar.ics]
        for key in [k for k in self._sources if k not in dict(sources)]:  # (taken out of the config since)
            del self._sources[key]
        errors, read = [], 0
        for key, reader in sources:
            try:
                self._sources[key] = await reader(start, end)
                read += 1
            except CalendarError as e:
                label = "Google" if key == "google" else key[4:]
                errors.append(f"{label}: {e}")
        error = "; ".join(errors)
        if error and error != self.error:
            log.warning("calendar: %s", error)
        self.error = error
        if not read and not self._sources:
            return
        self.cals = [c for cals, _ in self._sources.values() for c in cals]
        self.cals.sort(key=lambda c: (not c.primary, c.kind != "google", c.name.lower()))
        self.events = sorted((e for _, evs in self._sources.values() for e in evs), key=lambda e: (e.starts(), e.cal_name))
        if read:
            self.window, self.synced = (start, end), time.time()
        await self._event_reminders()

    async def _read_google(self, start: datetime, end: datetime) -> tuple[list[Cal], list[Event]]:
        cals = await self.fetch_calendars(self.cfg, self.google)
        return cals, await self._fetch(start, end, cals)

    async def _read_feed(self, name: str, start: datetime, end: datetime) -> tuple[list[Cal], list[Event]]:
        f = self.feed(name)
        await f.refresh(self.cfg.calendar.ics_minutes * 60)
        cal = f.cal()
        return [cal], await self._fetch(start, end, [cal])

    async def _fetch(self, start: datetime, end: datetime, cals: list[Cal] | None = None) -> list[Event]:
        out = []
        for cal in cals or self.cals:
            if cal.kind == "ics":
                evs = await asyncio.to_thread(self.feed(cal.name).events, cal, start, end)
            else:
                evs = [ev for item in await self.google.events(cal.id, start, end) if (ev := make_event(cal, item))]
            for ev in evs:
                out.append(ev)
                self.ids[ev.sid] = (ev.cal, ev.id)
        return sorted(out, key=lambda e: (e.starts(), e.cal_name))

    async def between(self, start: datetime, end: datetime, cal: str = "", search: str = "") -> list[Event]:
        """Events overlapping [start, end): from memory when that's within what was read lately."""
        if not self.cals or time.time() - self.synced > 3600:
            await self.sync()
            if self.error:
                raise CalendarError(self.error)
        cals = [self.cal(cal)] if cal else self.cals
        fresh = self.window and self.window[0] <= start and end <= self.window[1] \
            and time.time() - self.synced < self.cfg.calendar.sync_minutes * 60 + 60
        events = self.events if fresh else await self._fetch(start, end, cals)
        ids = {c.id for c in cals}
        out = [e for e in events if e.cal in ids and e.starts() < end and e.ends() > start]
        if search:
            words = search.lower().split()
            out = [e for e in out if all(w in f"{e.title} {e.where} {e.note}".lower() for w in words)]
        return out

    def cached_day(self, day: date) -> list[Event]:
        start = local(datetime.combine(day, dtime(0, 0)))
        end = start + timedelta(days=1)
        return [e for e in self.events if e.starts() < end and e.ends() > start]

    @staticmethod
    def busy(events: list[Event]) -> list[Event]:
        return [e for e in events if not e.all_day and not e.declined and not e.free]

    def free_slots(self, events: list[Event], start: datetime, end: datetime, minimum: int = 30 * 60
                   ) -> list[tuple[datetime, datetime]]:
        slots, at = [], start
        for e in sorted(self.busy(events), key=lambda e: e.starts()):
            if e.starts() > at and (e.starts() - at).total_seconds() >= minimum:
                slots.append((at, min(e.starts(), end)))
            at = max(at, e.ends())
            if at >= end:
                break
        if (end - at).total_seconds() >= minimum:
            slots.append((at, end))
        return [(a, b) for a, b in slots if b > a]

    def find(self, sid: str) -> tuple[str, str]:
        hit = self.ids.get(sid.strip().lower())
        if not hit:
            raise CalendarError(f"no event {sid!r}: list the events again (ov cal list) for their ids")
        return hit

    # ---- the agent's context
    def context(self) -> str:
        if not self.enabled:
            return ""
        if not self.synced:
            return f"<calendar>not read yet{': ' + self.error if self.error else ''}</calendar>"
        now = local(datetime.now())
        today = now.date()
        lines = []
        for day, label in ((today, "Today"), (today + timedelta(days=1), "Tomorrow")):
            evs = self.cached_day(day)
            if not evs:
                lines.append(f"{label}: nothing in the calendar")
                continue
            busy = sum((min(e.ends(), local(datetime.combine(day, dtime(23, 59)))) -
                        max(e.starts(), local(datetime.combine(day, dtime(0, 0))))).total_seconds()
                       for e in self.busy(evs))
            lines.append(f"{label} ({len(evs)} event{'s' if len(evs) != 1 else ''}, {duration_text(busy)} busy):")
            for e in evs:
                mark = " (now)" if e.starts() <= now < e.ends() and not e.all_day else \
                    " (over)" if e.ends() <= now and not e.all_day else ""
                lines.append(f"- {e.line(today, show_cal=len(self.cals) > 1)}{mark}")
        stale = f" (last read {duration_text(time.time() - self.synced)} ago: {self.error})" if self.error else ""
        return "<calendar>" + stale + "\n" + "\n".join(lines) + "\n</calendar>"

    # ---- reminders before events
    async def _event_reminders(self) -> None:
        minutes = self.cfg.calendar.event_reminders
        rem = self.domain.reminders
        have: dict[str, str] = json.loads(self.store.get(REMINDERS_KEY) or "{}")
        now = time.time()
        want = {}
        if minutes:
            horizon = now + minutes * 60 + self.cfg.calendar.sync_minutes * 60 * 2
            for e in self.busy(self.events):
                at = e.starts().timestamp() - minutes * 60
                if now < e.starts().timestamp() and at <= horizon:
                    want[f"{e.sid}@{int(e.starts().timestamp())}"] = (max(at, now), e)
        pending = {r["id"] for r in rem.pending()}
        for key, rid in list(have.items()):
            if key not in want:
                if rid in pending:
                    rem.remove(rid)  # (moved or cancelled since)
                del have[key]
        for key, (at, e) in want.items():
            if key not in have:
                text = f"📅 {e.title or 'An event'} at {e.starts():%H:%M}" + (f" · {e.where}" if e.where else "")
                have[key] = rem.add(at, text, kind="reminder")["id"]
        self.store.set(REMINDERS_KEY, json.dumps(have))

    async def loop(self) -> None:
        while True:
            if self.enabled:
                try:
                    await self.sync()
                except Exception:
                    log.exception("calendar")
            await asyncio.sleep(self.cfg.calendar.sync_minutes * 60)

    # ---- changes
    async def _get(self, cal: str, eid: str) -> dict:
        return await self.google.call("GET", f"/calendars/{_q(cal)}/events/{_q(eid)}")

    async def _existing(self, sid: str) -> tuple[Cal, Event]:
        cal_id, eid = self.find(sid)
        cal = next((c for c in self.cals if c.id == cal_id), None)
        if not cal:
            raise CalendarError("that calendar isn't one it sees any more")
        if cal.kind != "google":
            raise CalendarError(f"{cal.name} is read-only (an iCal link)")
        if cal.mode == "read":
            raise CalendarError(f"{cal.name} is read-only for you (calendar.ask / calendar.auto in the config)")
        ev = make_event(cal, await self._get(cal_id, eid))
        if not ev:
            raise CalendarError("that event was deleted")
        if ev.others:
            raise CalendarError(f"“{ev.title}” has other people invited: Overseer leaves those to them (change it in "
                                "Google Calendar)")
        if cal.mode == "read":
            raise CalendarError(f"{cal.name} is read-only for you (calendar.ask / calendar.auto in the config)")
        return cal, ev

    async def propose_add(self, session, title: str, start, end, all_day: bool, cal_name: str = "",
                          where: str = "", note: str = "") -> str:
        cal = self.cal(cal_name) if cal_name else self.default_cal()
        if cal.kind != "google":
            raise CalendarError(f"{cal.name} is read-only (an iCal link)")
        if cal.mode == "read":
            raise CalendarError(f"{cal.name} is read-only for you (calendar.ask / calendar.auto in the config)")
        tz = self.cfg.local.timezone or ""
        body = {"summary": title, "start": _google_time(start, tz), "end": _google_time(end, tz)}
        if where:
            body["location"] = where
        if note:
            body["description"] = note
        return await self._propose(session, {"action": "add", "cal": cal.id, "body": body}, cal)

    async def propose_change(self, session, sid: str, *, start=None, end=None, title=None, where=None,
                             note=None) -> str:
        cal, ev = await self._existing(sid)
        tz = self.cfg.local.timezone or ""
        body: dict = {}
        if start is not None:
            body["start"], body["end"] = _google_time(start, tz), _google_time(end, tz)
        for k, v in (("summary", title), ("location", where), ("description", note)):
            if v is not None:
                body[k] = v
        if not body:
            raise CalendarError("nothing to change: give a new time, --title, --where or --note")
        old = {k: ev.raw.get(k, "" if k in ("summary", "location", "description") else None) for k in body}
        return await self._propose(session, {"action": "change", "cal": cal.id, "event": ev.id, "body": body,
                                             "old": old, "title": ev.title}, cal)

    async def propose_delete(self, session, sid: str) -> str:
        cal, ev = await self._existing(sid)
        old = {k: ev.raw[k] for k in COPY_FIELDS if k in ev.raw}
        return await self._propose(session, {"action": "delete", "cal": cal.id, "event": ev.id, "old": old,
                                             "title": ev.title}, cal)

    def describe(self, p: dict, cal: Cal | None = None, html: bool = True) -> str:
        e = esc if html else (lambda s: s)
        name = (cal or next((c for c in self.cals if c.id == p["cal"]), None) or Cal(p["cal"], p["cal"])).name
        if p["action"] == "add":
            b = p["body"]
            s, en = _span(b)
            return (f"{e(b.get('summary', ''))} · {e(span_text(*s, en))} in {e(name)}"
                    + (f"\n📍 {e(b['location'])}" if b.get("location") else ""))
        if p["action"] == "delete":
            s, en = _span(p["old"])
            return f"{e(p['title'])} · {e(span_text(*s, en))} from {e(name)}"
        parts = []
        b, old = p["body"], p["old"]
        if "start" in b:
            s0, a0 = _span(old)
            s1, a1 = _span(b)
            parts.append(f"{e(span_text(*s0, a0))} → {e(span_text(*s1, a1))}")
        for k, label in (("summary", "title"), ("location", "place"), ("description", "note")):
            if k in b:
                parts.append(f"{label}: {e(str(old.get(k) or '-')[:80])} → {e(str(b[k] or '-')[:80])}")
        return f"{e(p['title'])} in {e(name)}: " + "; ".join(parts)

    async def _propose(self, session, p: dict, cal: Cal) -> str:
        pid = secrets.token_hex(4)
        p.update(session=getattr(session, "id", None), at=time.time())
        verb = {"add": "Add", "change": "Change", "delete": "Delete"}[p["action"]]
        if cal.mode == "auto":
            done = await self._apply(p)
            self.store.set(KEY + pid, json.dumps({**done, "state": "done"}))
            await self.domain.notifier.send(Outgoing(
                f"📅 {self._past(p)}: {self.describe(done, cal)}", session_id=p["session"], silent=True,
                buttons=[[("↩️ Undo", f"calu:{pid}")]]))
            return f"✓ {self._past(p).lower()} in {cal.name} (they see it, with ↩️ Undo): {self.describe(done, cal, False)}"
        self.store.set(KEY + pid, json.dumps({**p, "state": "asked"}))
        await self.domain.notifier.send(Outgoing(
            f"📅 <b>{verb}?</b> {self.describe(p, cal)}", session_id=p["session"],
            buttons=[[(f"✅ {verb}", f"cal:{pid}:ok"), ("✖ No", f"cal:{pid}:no")]]))
        return (f"⏳ asked them (a card with ✅ {verb} / ✖ No): nothing changes until they tap; their answer reaches you "
                "in your next turn")

    @staticmethod
    def _past(p: dict) -> str:
        return {"add": "Added", "change": "Changed", "delete": "Deleted"}[p["action"]]

    async def _apply(self, p: dict) -> dict:
        """Make the change in Google; the record of it (for ↩️ Undo)."""
        cal, path = p["cal"], f"/calendars/{_q(p['cal'])}/events"
        if p["action"] == "add":
            item = await self.google.call("POST", path, {"sendUpdates": "none"}, p["body"])
            p = {**p, "event": item["id"]}
        elif p["action"] == "change":
            current = await self._get(cal, p["event"])
            if make_event(Cal(cal, cal), current).others:
                raise CalendarError("it has other people invited now: change it in Google Calendar")
            p = {**p, "old": {k: current.get(k, "" if k in ("summary", "location", "description") else None)
                              for k in p["body"]}}
            await self.google.call("PATCH", f"{path}/{_q(p['event'])}", {"sendUpdates": "none"}, p["body"])
        else:
            await self.google.call("DELETE", f"{path}/{_q(p['event'])}", {"sendUpdates": "none"})
        self.domain.schedule_sync()
        return p

    async def answer(self, pid: str, ok: bool) -> tuple[str, list]:
        """✅ / ✖ on a card: (the card's new text, its buttons)."""
        raw = self.store.get(KEY + pid)
        p = json.loads(raw) if raw else None
        if not p or p.get("state") != "asked" or time.time() - p.get("at", 0) > KEEP_SECONDS:
            return "This was already answered (or is too old).", []
        verb = {"add": "Add", "change": "Change", "delete": "Delete"}[p["action"]]
        if not ok:
            self.store.set(KEY + pid, json.dumps({**p, "state": "declined"}))
            self._tell(p, f"They said no to your calendar change: {verb.lower()} {self.describe(p, html=False)}.")
            return f"✖ <s>{verb}: {self.describe(p)}</s>", []
        try:
            done = await self._apply(p)
        except CalendarError as e:
            self._tell(p, f"Your calendar change ({verb.lower()} {self.describe(p, html=False)}) failed: {e}")
            return f"⚠️ {verb}: {self.describe(p)}\n<i>{esc(e)}</i>", []
        self.store.set(KEY + pid, json.dumps({**done, "state": "done"}))
        self._tell(p, f"They approved, and it's done: {self._past(p).lower()} {self.describe(done, html=False)}.")
        return f"📅 {self._past(p)}: {self.describe(done)}", [[("↩️ Undo", f"calu:{pid}")]]

    async def undo(self, pid: str) -> str:
        raw = self.store.get(KEY + pid)
        p = json.loads(raw) if raw else None
        if not p or p.get("state") != "done" or time.time() - p.get("at", 0) > KEEP_SECONDS:
            return "This can't be undone any more."
        path = f"/calendars/{_q(p['cal'])}/events"
        what = self.describe(p)
        if p["action"] == "add":
            await self.google.call("DELETE", f"{path}/{_q(p['event'])}", {"sendUpdates": "none"})
        elif p["action"] == "change":
            await self.google.call("PATCH", f"{path}/{_q(p['event'])}", {"sendUpdates": "none"},
                                   {k: v for k, v in p["old"].items() if v is not None})
        else:
            await self.google.call("POST", path, {"sendUpdates": "none"}, p["old"])
        self.store.set(KEY + pid, json.dumps({**p, "state": "undone"}))
        self._tell(p, f"They undid your calendar change: {self._past(p).lower()} {self.describe(p, html=False)}.")
        self.domain.schedule_sync()
        return f"↩️ <s>{self._past(p)}: {what}</s>\n<i>undone</i>"

    def _tell(self, p: dict, text: str) -> None:
        agent = self.domain.agent
        if agent and p.get("session"):
            agent.add_note(p["session"], text)


def _span(body: dict) -> tuple[tuple, bool]:
    """((start, end), all_day) of an event body (Google's format)."""
    start, all_day = _parse_time(body["start"])
    end, _ = _parse_time(body.get("end") or body["start"])
    return (start, end), all_day
