"""`overseer config`: an old config file brought up to date with the current example, meaning the same."""

import re
from pathlib import Path

import pytest
import yaml

from overseer.__main__ import main
from overseer.core.config import Config, ConfigError, _build
from overseer.core.configtool import update

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("example", ["config.example.yaml", "assistant.example.yaml"])
def test_the_examples_are_up_to_date_with_themselves(tmp_path, example):
    f = tmp_path / "config.yaml"
    f.write_text((ROOT / example).read_text())
    report, text = update(f)
    assert text == f.read_text(), "an unchanged example must come back unchanged"
    assert "In your file but not used any more" not in report


def test_an_old_assistant_config(tmp_path, capsys):
    s = (ROOT / "assistant.example.yaml").read_text()
    s = re.sub(r"\n  timezone:.*", "", s)  # (settings that came later)
    s = re.sub(r"\n# Pictures, GIFs.*?giphy_rating: pg-13\n", "\n", s, flags=re.S)
    s = s.replace("/home/you/Vault", "/home/me/Vault").replace("[123456789]", "[5555]")
    s = s.replace("timeout_minutes: 30", "timeout_minutes: 45")
    s = s.replace("  live_progress: true", "  live_progress: true\n  job_reports: true")  # (no longer used)
    s = s.replace("    model: null\n    home: ./state/assistant/claude",
                  "    model: opus\n    pass_env: [HTTPS_PROXY]\n    home: ./state/assistant/claude")
    old = tmp_path / "assistant.yaml"
    old.write_text(s)
    old.chmod(0o640)
    assert main(["config", "-c", str(old)]) == 0
    out = capsys.readouterr().out
    assert "agent.timeout_minutes: 45   (default: 60)" in out and "vault.path: /home/me/Vault" in out
    assert "local.timezone: \"\"   [Asia/Dubai]" in out and "media.fetch: true" in out
    assert "agent.job_reports: true" in out.split("not used any more")[1]
    new = tmp_path / "assistant.yaml.new"
    assert old.read_text() == s and new.stat().st_mode & 0o777 == 0o640  # (yours untouched; same permissions)
    text = new.read_text()
    assert "# Pictures, GIFs and videos from the internet" in text and 'timezone: ""' in text
    assert "    model: opus" in text and '    pass_env: ["HTTPS_PROXY"]' in text and "job_reports" not in text
    clean = yaml.safe_load(s)
    clean["agent"].pop("job_reports")
    assert _build(Config, yaml.safe_load(text), "") == _build(Config, clean, "")  # it means the same
    # Run again on the new file: up to date.
    new.rename(old)
    assert main(["config", "-c", str(old)]) == 0 and "up to date" in capsys.readouterr().out


def test_an_old_hpc_config_with_projects(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("""ssh:
  hostname: login.cluster.edu
  user: ab123
slurm:
  poll_interval: 30
  compute: {partition: gpu, cpus: 4}
telegram: {allowed_chat_ids: [1, 2]}
projects:
  dreamer:
    dir: /scratch/ab123/dreamer
    context: "DreamerV3: crafter"
""")
    report, text = update(f)
    merged = yaml.safe_load(text)
    assert merged["projects"]["dreamer"]["context"] == "DreamerV3: crafter"
    assert merged["slurm"]["compute"]["partition"] == "gpu" and merged["slurm"]["compute"]["cpus"] == 4
    assert merged["slurm"]["compute"]["mem"] == "32G"  # (the default, written out)
    assert "slurm.poll_interval: 30   (default: 60)" in report
    assert _build(Config, merged, "") == _build(Config, yaml.safe_load(f.read_text()), "")
    assert "# Projects: each gets its own Telegram topic" in text


def test_a_broken_file_is_refused(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError):
        update(f)


def test_a_broken_file_says_where(tmp_path):
    from overseer.core.config import ConfigError, load_config
    p = tmp_path / "assistant.yaml"
    p.write_text("domain: assistant\nmedia:\n  fetch: true\ncalendar: {}\n  ics: {Uni: UNI_ICS_URL}\n")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "line 5" in str(e.value) and "   5 |   ics: {Uni: UNI_ICS_URL}" in str(e.value)
    assert "under its section's line" in str(e.value)
    p.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="isn't a YAML mapping"):
        load_config(p)
