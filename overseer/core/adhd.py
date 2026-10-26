"""The i-have-adhd output style for every agent of a bot (agent.i_have_adhd).

https://github.com/ayghri/i-have-adhd (MIT, by Ayoub Ghriss; copied from commit 1ecea19 into
skills/i-have-adhd/, with its license): rules that shape replies for a reader with ADHD - the next action first,
numbered steps, the state restated every turn, no tangents, specific time estimates, visible wins. In Claude Code
it's a skill the user turns on with /i-have-adhd, or always on through a session hook; Overseer's agents are
reached through Telegram, so it's always on here, the way its hook does it: the rules go into each agent's
instructions.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL = Path(__file__).parent / "skills" / "i-have-adhd" / "SKILL.md"
SOURCE = "https://github.com/ayghri/i-have-adhd"

INTRO = """

## Output style: i-have-adhd (always on)
The user turned on this output style for every agent in Overseer's config (agent.i_have_adhd; from \
{source}). Your replies reach them on Telegram, read on a phone: shape them by these rules. Your instructions above \
win where they conflict (its rule "A rule fights the harness"). If they say "stop adhd mode" or "normal mode", \
return to your default style for this conversation.
"""


def rules() -> str:
    """The skill's text, without its frontmatter, headings one level down (under the section above)."""
    text = SKILL.read_text()
    if text.startswith("---") and (end := text.find("\n---", 3)) > 0:
        text = text[end + 4:]
    text = re.sub(r"^# i-have-adhd\s*\n", "", text.strip(), flags=re.M)
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", text, flags=re.M).strip()


def section() -> str:
    return INTRO.format(source=SOURCE) + "\n" + rules()
