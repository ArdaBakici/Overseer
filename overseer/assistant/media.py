"""Media from the internet for the assistant's replies: `ov fetch <url>` and `ov gif <query>` (Giphy).

The agent's shell has no internet, so Overseer downloads for it - carefully, since Overseer isn't sandboxed:
- http(s) only, to public addresses only: the host is resolved first, every address must be a public one (not
  this machine, the LAN, the tailnet, link-local, ...), and the connection goes to that checked address (so the
  name can't be re-pointed in between); each redirect is checked the same way.
- images, GIFs and videos only, recognised by their first bytes (not by name or the server's say-so);
- at most media.max_mb, read in chunks and cut off past it.
The file lands in the conversation's ./outbox (sent with the reply) or ./inbox (to work on).
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import httpx

MAX_REDIRECTS = 4
TIMEOUT = 30.0
USER_AGENT = "Overseer/1 (personal assistant media fetch)"


class FetchError(ValueError):
    pass


@dataclass
class Fetched:
    path: Path
    kind: str  # png | jpeg | gif | webp | mp4 | webm
    size: int
    url: str  # where it finally came from


def sniff(head: bytes) -> str | None:
    """The file's type from its first bytes: only what Telegram shows as a picture, GIF or video."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    return None


EXT = {"png": ".png", "jpeg": ".jpg", "gif": ".gif", "webp": ".webp", "mp4": ".mp4", "webm": ".webm"}


def public_addresses(host: str, port: int, resolve=socket.getaddrinfo) -> list[str]:
    """The host's addresses, if all of them are public ones; else FetchError."""
    try:
        infos = resolve(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:
        raise FetchError(f"can't resolve {host}: {e}") from None
    addrs = list(dict.fromkeys(i[4][0] for i in infos))
    if not addrs:
        raise FetchError(f"{host} has no address")
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise FetchError(f"{host} points to a private address ({ip}): only public internet addresses")
    return addrs


def safe_name(url: str, kind: str, name: str = "") -> str:
    base = name or Path(urlsplit(url).path).stem or "download"
    base = re.sub(r"[^\w.-]+", "-", base).strip("-.")[:60] or "download"
    base = re.sub(r"\.(png|jpe?g|gif|webp|mp4|webm|mov)$", "", base, flags=re.I)
    return base + EXT[kind]


async def fetch(url: str, folder: Path, max_bytes: int, name: str = "", *, resolve=socket.getaddrinfo,
                transport: httpx.AsyncBaseTransport | None = None) -> Fetched:
    """Download one picture, GIF or video into `folder` (see the module's notes on what's allowed)."""
    folder.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False, transport=transport,
                                 headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise FetchError("only http:// and https:// addresses")
            if parts.username or parts.password:
                raise FetchError("no user names or passwords in the address")
            port = parts.port or (443 if parts.scheme == "https" else 80)
            addr = (await asyncio.to_thread(public_addresses, parts.hostname, port, resolve))[0]
            ip_host = f"[{addr}]" if ":" in addr else addr
            pinned = parts._replace(netloc=f"{ip_host}:{port}").geturl()
            req = client.build_request("GET", pinned, headers={"Host": parts.netloc.rsplit("@", 1)[-1]},
                                       extensions={"sni_hostname": parts.hostname})
            resp = await client.send(req, stream=True)
            try:
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    url = urljoin(url, resp.headers["location"])
                    continue
                if resp.status_code != 200:
                    raise FetchError(f"the server answered {resp.status_code}")
                declared = int(resp.headers.get("content-length") or 0)
                if declared > max_bytes:
                    raise FetchError(f"it's {declared / 2**20:.1f} MB; the limit is {max_bytes / 2**20:.0f} MB")
                data = bytearray()
                async for chunk in resp.aiter_bytes():
                    data += chunk
                    if len(data) > max_bytes:
                        raise FetchError(f"it's over the {max_bytes / 2**20:.0f} MB limit")
                kind = sniff(bytes(data[:16]))
                if kind is None:
                    raise FetchError("that isn't a picture, GIF or video (only those can be fetched)")
                path = folder / safe_name(url, kind, name)
                n = 1
                while path.exists():
                    path = path.with_name(f"{path.stem}-{n}{path.suffix}")
                    n += 1
                path.write_bytes(bytes(data))
                return Fetched(path, kind, len(data), url)
            finally:
                await resp.aclose()
        raise FetchError("too many redirects")


async def giphy_search(query: str, key: str, rating: str = "pg-13", limit: int = 8,
                       transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    """Giphy's search: [{"title", "url" (a GIF of at most ~2 MB), "page"}]."""
    api = (f"https://api.giphy.com/v1/gifs/search?api_key={quote(key)}&q={quote(query)}&limit={limit}"
           f"&rating={quote(rating)}&lang=en")
    async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as client:
        try:
            resp = await client.get(api)
        except httpx.HTTPError as e:
            raise FetchError(f"Giphy didn't answer: {type(e).__name__}") from None
    if resp.status_code != 200:
        raise FetchError(f"Giphy answered {resp.status_code}" + (" (is the API key right?)"
                                                                 if resp.status_code in (401, 403) else ""))
    out = []
    for g in (resp.json().get("data") or []):
        images = g.get("images") or {}
        best = next((images[k]["url"] for k in ("downsized", "downsized_medium", "fixed_height", "original")
                     if k in images and images[k].get("url")), None)
        if best:
            out.append({"title": g.get("title") or g.get("slug") or "gif", "url": best, "page": g.get("url", "")})
    return out


@functools.lru_cache(maxsize=1)
def local_tools() -> dict[str, bool]:
    """Image and video tools on this machine (the agent's sandbox can use what's installed here)."""
    def py_has(mod: str) -> bool:
        py = shutil.which("python3")
        if not py:
            return False
        try:
            return subprocess.run([py, "-c", f"import {mod}"], capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return {"ffmpeg": bool(shutil.which("ffmpeg")), "imagemagick": bool(shutil.which("magick") or shutil.which("convert")),
            "python3 Pillow": py_has("PIL")}


def capabilities(cfg, giphy_on: bool) -> str:
    tools = local_tools()
    have = ", ".join(n for n, ok in tools.items() if ok) or "none"
    missing = ", ".join(n for n, ok in tools.items() if not ok)
    return f"""What you can send and do here (Telegram, through Overseer):

Sending with your reply - files you put in ./outbox/ (up to 10 per reply):
  .png .jpg .webp (up to 10 MB)  -> pictures
  .gif                           -> an animation (plays in the chat)
  .mp4 .mov .m4v .webm           -> a video (plays in the chat)
  anything else (up to 50 MB)    -> a document
Getting media from the internet (your shell has no internet; WebSearch/WebFetch read pages as text):
  ov fetch <url>                 {"on" if cfg.media.fetch else "OFF (media.fetch)"}: pictures, GIFs, videos up to {cfg.media.max_mb} MB from public addresses
  ov gif <query>                 {"on (Giphy)" if giphy_on else "OFF (no Giphy key configured)"}
Making or editing images and videos locally: installed here: {have}{f"; not installed: {missing}" if missing else ""}
Reacting: ov react <emoji> on the user's last message - Telegram allows only its reaction set (👍 👎 ❤ 🔥 🥰 👏 😁 🤔
  🤯 😱 😢 🎉 🤩 🙏 👌 😍 💯 🤣 ⚡ 🏆 💔 🤨 😐 😴 😭 🤓 👀 🙈 😇 🤝 ✍ 🤗 🫡 💅 🤪 🗿 🆒 😘 💊 😎 🤷 ...)
Buttons: on the messages Overseer sends for you - reminders, check-ins (ov commit), focus sessions, `say:` routines.
Over time: routines (Assistant/Routines.md), ov remind / ov commit (come back until answered), ov wake (wakes you).
Markdown in your reply: bold, italics, lists, `code`, links, small tables (read on a phone).
Not possible: stickers, voice messages, polls, editing or deleting messages already sent, writing to anyone but
  the user, internet from your shell."""
