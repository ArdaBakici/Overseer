# The personal assistant: a second bot

The same platform runs a second, separate bot: a daily assistant whose memory is your **Obsidian vault**. It runs
from Overseer's folder next to the HPC bot, as its own process with its own Telegram bot, config, state and
logins, and it can't reach anything of the HPC bot's.

## Setting it up

Set up the HPC bot's folder first ([setup.md](setup.md), steps 1-2: the packages, the CLIs and Overseer itself); the
assistant doesn't need SSH or an HPC. Then:

1. **Its own Telegram bot.** Create a second bot with @BotFather, exactly as in [setup step 3](setup.md#3-create-your-telegram-bot),
   and add its token to `.env` under its own name: `ASSISTANT_BOT_TOKEN=123456789:...`.
2. **Its config.** `cp assistant.example.yaml assistant.yaml`, then set `vault.path` (your Obsidian vault on this
   machine, e.g. the folder Syncthing keeps in sync), `telegram.allowed_chat_ids` and `local.timezone`.
3. **Its own logins** (separate from the HPC bot's), then check and run it:

```bash
uv run overseer login claude_code -c assistant.yaml    # its own login (./state/assistant/claude); and/or codex
uv run overseer check -c assistant.yaml
uv run overseer run -c assistant.yaml
```

4. **Optional:** your calendars ([below](#your-calendars-optional)) and topics ([below](#topics-media-and-the-rest)).

To keep it running, copy `deploy/overseer.service` to `overseer-assistant.service` and change `config.yaml` to
`assistant.yaml` in its `ExecStart` line ([setup step 8](setup.md#8-keep-it-running-with-systemd)).

## What it does

- **The vault is its memory.** Point `vault.path` at the folder Syncthing keeps in sync on the homelab. The agent
  reads and edits notes there with real commands (`rg`, editing files): your daily note, notes it keeps in
  `Assistant/Memory/`, anything you ask about. It follows Obsidian's conventions ([[links]], #tags) and never
  writes `.obsidian/` (Obsidian's settings) or Syncthing's own files: the sandbox refuses it.
- **Personality: yours to give.** On its first run it just says hi and asks you who it should be - a name, how
  to talk and behave - and writes that into `Assistant/Persona.md` (edit it in Obsidian, or ask it to change).
  `Assistant/About me.md` is what it knows about you; it adds lasting facts there and tells you. Both go into its
  instructions on every reply. `/persona` shows it. It also keeps two notes about you for other AI agents:
  `Assistant/My writing style.md` (informal and formal, written so another model can imitate you) and
  `Assistant/My personality.md`.
- **↩️ Undo.** Overseer keeps a history of the vault in a git repository in the bot's state folder, outside the
  vault (so nothing extra is synced and the agent can't touch it). Before each reply it records your edits since,
  after it the reply's changes; a reply that changed notes lists them with a ↩️ Undo button, which reverts exactly
  those changes and keeps yours (if you edited the same lines since, it says so and changes nothing). `/undo`
  lists recent changes, `/vault` shows the history and Syncthing's conflict copies.
- **Tasks** are the Obsidian Tasks plugin's checklist lines, anywhere in the vault (`- [ ] Call the bank 📅
  2026-10-20 ⏫ #home`, with 🔁 recurrences). `/tasks [today|week|#tag|words]` lists them with ✅ buttons,
  `/task <text>` adds one; the agent uses `ov task list|add|done`. Completing a recurring task adds its next
  occurrence, like the plugin. New tasks go to `Assistant/Tasks.md` unless it belongs in a particular note.
- **It builds the system with you.** Nothing is set up in advance: you just talk to it, and it suggests one thing at
  a time when it notices a pattern ("a daily reminder for that at 9:30?"), with tap buttons (`ov ask`). How it
  organizes your things is its own note, `Assistant/System.md`, which it keeps up to date. Its tools:
  - **Routines** in `Assistant/Routines.md`, added with your agreement: a `do:` routine wakes the assistant at its
    time (it may decide there's nothing worth saying and stay silent); a `say:` one ("Meds 💊") just sends its
    message. You change them by asking ("move it to 22:00", "pause it until I'm back").
  - **Reminders** fire once, with ✅ Done · 💤 Later · ⏭ Skip; ones you want to come back until answered get a
    follow-up policy (the note's `# Follow-ups`, e.g. "again after 10m, up to 5 more times").
  - **Check-backs**: when it thinks one would help ("I'll send it by 3"), it asks first, then checks in at 3
    ("how did it go?" ✅ · 💤 · 🧱 Stuck - which brings it in to help).
  - **Someday list**: things to do sometime ("learn to draw") are tasks tagged `#someday` in `Assistant/Someday.md`;
    it brings one up now and then on a day that looks light - its judgement, not Overseer's.
  - **Focus sessions**: "let's do 25 minutes on the intro" - one quiet countdown message, then a ping.
  - `/remind` and `/reminders` still exist as shortcuts.
### Your calendars (optional)

- **Your calendars** (optional): Google Calendar, and read-only calendars from iCal links. The assistant sees
  today's and tomorrow's events in every request and looks
  further with `ov cal`. It plans around them: it checks free time before suggesting when to do something, it
  reads a packed day as "smaller and fewer suggestions" and a light one as room for a bigger task or a someday
  item, and it can offer to block time for a task. It changes events only where you allow it: in calendars under
  `calendar.ask` you get a card (✅ Add / ✖ No) and nothing happens until you tap; in ones under `calendar.auto`
  it's done at once and the notice has ↩️ Undo. It never invites anyone and leaves events with other people alone.
  `calendar.event_reminders: 10` adds a reminder 10 minutes before each timed event.

  **Planning with the calendar** (time-blocking, a day or week laid out, re-planning) isn't a fixed method: the
  assistant has tools and works out with you what helps, keeping what works in `Assistant/System.md`. It can send
  several changes as **one proposal** (one card, ✅ Apply all / ✖ No - answer in words to adjust it; one ↩️ Undo
  for all of it; refused before it reaches you if it overlaps busy events), **label** its own events and **link**
  them to Obsidian tasks, and tie **actions to events**: "when the intro block ends, ask me if it's done" (✅ ticks
  the task off) or "10 minutes before the exam, wake up and help me get ready". Actions follow an event when you
  move it in Google Calendar, go when it's deleted, and work on any event, iCal ones included. The Google login stays with
  Overseer (in the bot's state folder, which agents can't read); the agent only uses `ov cal`. Set up:
  1. In [Google Cloud](https://console.cloud.google.com): a project, enable the **Google Calendar API**; the
     **OAuth consent screen**: External, add yourself as a test user, then **publish** it (left in "Testing",
     Google drops the login after 7 days; unverified is fine for just you); **Credentials → OAuth client ID →
     Desktop app**.
  2. Its id and secret into `.env`: `GOOGLE_CLIENT_ID=...` and `GOOGLE_CLIENT_SECRET=...`.
  3. `uv run overseer login google -c assistant.yaml`: open the link on any device, allow access; the browser then
     fails to load a `127.0.0.1` page - copy that page's address back into the terminal. It lists your calendars
     and what the assistant may do with each.
  4. `calendar.enabled: true` (and `ask` / `auto` as you like), restart the bot. `overseer check` shows it.

  **iCal links** are read-only calendars it sees and plans around but never changes: a university timetable, a
  calendar someone shared, Outlook's or Apple's "publish calendar" link, a holidays feed. List them under
  `calendar.ics` by name; the value is the link (`https://` or `webcal://`), or - since such links usually work
  like a password - the name of a variable in `.env` holding it: `ics: {Uni: UNI_ICS_URL}` and
  `UNI_ICS_URL=https://...` in `.env`. Overseer fetches each one every `ics_minutes` (30) and works out repeating
  events, their exceptions and time zones itself. They work without Google too (`enabled: false`).

### Topics, media and the rest

- **Topics** (optional): set `telegram.topics: true` and switch on threaded mode for the bot in @BotFather.
  Everyday talk stays in Telegram's "All" view itself (`telegram.general: all`, the example's choice): no general
  topic. Overseer makes one topic per project, a lasting subject such as a trip or a move (`/project new japan Trip
  in April`, or the assistant creates one with `ov project new`, optionally with a notes folder in the vault), plus
  🛠 feedback and 🌐 web. In @BotFather (your bot → Bot Settings → Threaded mode), turn off users creating topics:
  otherwise Telegram turns each message you write in "All" into a topic of its own (Overseer then moves it into
  "All" and tells you once at startup). Switching to `all` deletes a 💬 general topic of before; the conversation
  itself carries on in "All". `general: topic` keeps a 💬 general topic, as the HPC bot does. With `topics: false`
  (the example's default) it's one plain chat.
- **Media and reactions.** `.gif` files in a reply play as animations and videos as videos. Its shell has no
  internet, so `ov fetch <url>` has Overseer download a picture, GIF or video for it: public addresses only (never
  this machine, the LAN or the tailnet; the address is checked and pinned, redirects too), recognised by its
  bytes, up to `media.max_mb`; each download is noted in a 🌐 web topic. `ov gif <query>` searches Giphy, if
  you give it a key (`media.giphy_key_env`). `ov react 🎉` reacts to your last message instead of replying.
  Image tools (Pillow, ffmpeg, ImageMagick) work in its sandbox if they're installed on the homelab;
  `ov capabilities` tells it what it can send and do.
- **Notes it always keeps in mind**: besides Persona and About me, any note with frontmatter `assistant: always`
  (e.g. a writing style it maintains for you); `assistant: index` lists one by name and description.
- **Its time zone** is `local.timezone` (e.g. Asia/Dubai), whatever the homelab's is; reminders confirm the exact
  moment (`Thu 2026-10-22 10:00 (+04:00, tomorrow)`).
- **Starting over.** `uv run overseer reset -c assistant.yaml` (with the bot stopped) lists and wipes its
  flags: pending reminders and check-ins, the day logs, the first-hello flag, wake-ups it set itself, paused
  wake-ups. Then it asks whether to forget the conversations too: their records, transcripts and search index,
  and the agent CLIs' own sessions - without them it can't resume or search old chats. Your notes, projects,
  settings and logins stay either way (`--yes` answers only the first question). For a completely fresh start,
  also delete the `Assistant/` folder in the vault: it then says hi and asks for a personality again.
- **Every morning**: today's events, today's tasks and the overdue ones, today's reminders, what changed in the vault. Its
  **dashboard** (its own port, on your tailnet like the HPC bot's) has Today, Tasks and Changes tabs. `/search`
  also searches the vault's notes.
### Kept apart from the HPC bot

- **Kept apart from the HPC bot.** On Linux the sandbox can't filter unix sockets by path, so each bot notes its
  folders in `./state/bots/` and every bot's agent is denied the others' state and tmp folders (the HPC bot's SSH
  connection socket is in its tmp folder), all of `./state`, and every `.env*` file. Its folders must be its own
  (`./state/assistant`, `/tmp/overseer-assistant`), and the vault must be outside Overseer's folder.
  The end-to-end test runs both bots side by side with the real CLIs and checks this.
