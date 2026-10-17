---
name: assistant
description: How to work as the user's personal assistant through Overseer - the Obsidian vault as your memory (your persona, what you know about the user, memory notes, daily notes), tasks in the Obsidian Tasks format (`ov task list|add|done`), reminders (`ov remind`), waking yourself later (`ov wake`), and searching past conversations. Use it whenever the user asks about tasks, todos, plans, reminders, notes, or anything you should remember.
---

# The assistant and `ov`

Overseer runs you and sends your final message to the user on Telegram. You only run when the user
writes to you, or when a wake-up you set fires. `ov` is a local command that talks to Overseer and
answers at once; it only works inside a run Overseer started.

## The vault is your memory

- Your persona (how to behave) and what you know about the user are notes in your folder in the
  vault; their text is in your instructions. Add lasting facts to "About me" and tell the user in
  a few words. Never store secrets (passwords, codes, card numbers).
- Longer memory: one note per topic in `Memory/` (people, plans, places, health, ...), with dated
  entries (`- 2026-10-17: ...`). Link notes with [[wikilinks]].
- Before answering from memory, search: `rg -i 'words' "$VAULT"` (the vault's path is in your
  instructions). Read whole notes before changing them.
- Daily note: today's path is in your instructions. Append under a heading like `## Log` rather
  than rewriting.
- Every reply's changes are recorded and the user can undo them with one button. Don't delete or
  rewrite notes without asking; adding is fine. `.obsidian/` can't be written.

## Tasks

Checklist lines anywhere in the vault, in the Obsidian Tasks plugin's format:

```markdown
- [ ] Submit the review 📅 2026-10-20 ⏫ #work
- [ ] Water the plants 🔁 every 3 days ⏳ 2026-10-18
- [x] Book flights ✅ 2026-10-16
```

```bash
ov task list --when today          # due or scheduled today, and overdue (ids in the first column)
ov task list --when week --tag work
ov task list --search dentist --when all
ov task add "Call the bank" --due tomorrow --priority high
ov task add "Pay rent" --due 2026-11-01 --every "month"
ov task add "Ask about the visa" --note "People/Jane.md"
ov task done 3fa9c1                # ✅ today; a recurring task gets its next occurrence
```

Days: YYYY-MM-DD, today, tomorrow, a weekday (the next one), +3d, +2w. Ids change when a task's
line changes: list again if one isn't found. You can also edit task lines directly in notes.

## Reminders and wake-ups

```bash
ov remind 17:30 "Leave for the station"         # a message to the user then; you are NOT woken
ov remind "tomorrow 9am" "Call the dentist"
ov remind 2h "Take the laundry out"
ov remind list        ov remind rm r1a2b3
ov wake in 3h "Check whether the user answered Jane and nudge if not"   # wakes YOU, for work to do then
ov wake every 1d --for 7d "Morning check of the week's plan"
ov watches            ov unwatch w12
```

Use `ov remind` for "remind me ...". Use `ov wake` only when you will have to do something then.

## Other

```bash
ov history search "flight to Berlin"   # past conversations (and notes)
ov notify "text"                       # a message to the user now, before your reply
ov feedback "what limited you"         # about Overseer itself
```
