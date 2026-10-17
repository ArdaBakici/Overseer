import json
from overseer.hpc.diagnose import extract_metrics, extract_traceback, find_issues, tqdm_cleanup
from overseer.core.notify import split_message
from overseer.hpc.slurm import (
    gpus_from_tres,
    normalize_state,
    parse_sacct_list,
    parse_sacct_steps,
    parse_scontrol,
    parse_squeue,
    summarize_steps,
)


def test_squeue():
    out = (
        "101|train|PENDING|gpu|0:00|1-00:00:00|1|(Priority)|gres/gpu:2|2026-10-07T10:00:00|N/A\n"
        "102|a|b|RUNNING|gpu|1:02:03|2:00:00|1|gpu-node-07|gres/gpu:a100:1|2026-10-07T09:00:00|2026-10-07T09:05:00\n"
    )
    jobs = parse_squeue(out)
    assert [j.job_id for j in jobs] == ["101", "102"]
    assert jobs[0].state == "PENDING" and jobs[0].reason == "(Priority)"
    assert jobs[1].name == "a|b"  # pipe in name survives
    assert jobs[1].reason == "gpu-node-07" and jobs[1].is_active


def test_sacct_list_and_states():
    out = (
        "200|ok|COMPLETED|0:0|01:00:00|02:00:00|s|e|gpu|n1|billing=1,gres/gpu=2|/home/me/p\n"
        "200.batch|batch|COMPLETED|0:0|01:00:00||s|e||n1||\n"
        "201|bad|CANCELLED by 1234|0:15|00:10:00|02:00:00|s|e|gpu|n2|gres/gpu:a100=1|/home/me/q\n"
    )
    jobs = parse_sacct_list(out)
    assert [j.job_id for j in jobs] == ["200", "201"]
    assert jobs[1].state == "CANCELLED" and jobs[1].is_terminal
    assert gpus_from_tres(jobs[0].alloc_tres) == "2"
    assert gpus_from_tres(jobs[1].alloc_tres) == "1"
    assert normalize_state("COMPLETING+") == "COMPLETING"


def test_steps_summary():
    out = (
        "300|train|OUT_OF_MEMORY|0:125|00:05:00||32G|billing=1|n1\n"
        "300.batch|batch|OUT_OF_MEMORY|0:125|00:05:00|33000000K||cpu=4|n1\n"
        "300.extern|extern|COMPLETED|0:0|00:05:00|1000K||cpu=4|n1\n"
    )
    s = summarize_steps(parse_sacct_steps(out))
    assert s["max_rss"].endswith("G")
    assert s["req_mem"] == "32G"
    assert "300.batch=OUT_OF_MEMORY" in s["failed_steps"]


def test_scontrol():
    line = ("JobId=5 JobName=my job UserId=me(1) JobState=RUNNING Command=/home/me/run.sh --lr 1e-3 "
            "WorkDir=/home/me/proj StdErr=/home/me/proj/slurm-5.out StdOut=/home/me/proj/slurm-5.out TRES=cpu=4,gres/gpu=1")
    d = parse_scontrol(line)
    assert d["JobName"] == "my job"
    assert d["Command"] == "/home/me/run.sh --lr 1e-3"
    assert d["StdOut"] == "/home/me/proj/slurm-5.out"
    assert d["TRES"] == "cpu=4,gres/gpu=1"


LOG = """epoch 1/10 loss: 2.31 acc=0.12
epoch 2/10 loss: 1.95 acc=0.25
Traceback (most recent call last):
  File "train.py", line 10, in <module>
    out = model(x)
  File "model.py", line 3, in forward
    return self.big(x)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
srun: error: gpu-07: task 0: Exited with exit code 1
"""


def test_findings_and_traceback():
    cats = [f.category for f in find_issues(LOG, "FAILED")]
    assert cats[0] == "GPU out of memory"
    tb = extract_traceback(LOG)
    assert tb.startswith("Traceback") and tb.endswith("Tried to allocate 2.00 GiB")
    assert [f.category for f in find_issues("", "TIMEOUT")] == ["Time limit"]


def test_metrics():
    m = extract_metrics(LOG)
    assert m["epoch"] == "2/10" and m["loss"] == "1.95" and m["acc"] == "0.25"
    assert tqdm_cleanup("10%|#\r50%|#####\r100%|##########") == "100%|##########"


def test_split_message_keeps_pre_balanced():
    text = "header\n<pre>" + "\n".join(f"line {i} " + "x" * 50 for i in range(300)) + "</pre>\nfooter"
    chunks = split_message(text, limit=1000)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 1000
        assert c.count("<pre>") == c.count("</pre>"), c[:80]


def test_markdown_to_telegram_html():
    from overseer.core.richtext import md_to_html
    out = md_to_html("## Result\nSeed 1 scored **0**: the `reward` wrapper. See [log](https://x.org/a?b=1&c=2).\n\n"
                     "1. fixed `a.py`\n2. job **42**\n   - lr `3e-4`\n\n```python\nr = 1 < 2\n```\n\n"
                     "| seed | score |\n|---|---|\n| 0 | 3.1 |\n\n> quote with `code` and [a](https://x.org)\n\n"
                     "Raw <b>html</b> & ~~old~~ [bad](javascript:alert(1))")
    assert out.startswith("<b>Result</b>\n\nSeed 1 scored <b>0</b>: the <code>reward</code> wrapper.")
    assert '<a href="https://x.org/a?b=1&amp;c=2">log</a>' in out
    assert "1. fixed <code>a.py</code>\n2. job <b>42</b>\n   • lr <code>3e-4</code>" in out
    assert '<pre><code class="language-python">r = 1 &lt; 2</code></pre>' in out
    assert "<pre>seed │ score\n─────┼──────\n0    │ 3.1</pre>" in out
    # Telegram can't nest code or links in a quote: they become plain text there.
    assert "<blockquote>quote with code and a</blockquote>" in out
    assert "Raw &lt;b&gt;html&lt;/b&gt; &amp; <s>old</s>" in out and '<a href="javascript' not in out


def test_split_html_keeps_tags_balanced():
    import re
    from overseer.core.richtext import split_html, strip_html
    text = ("<b>head</b>\n<blockquote expandable>" + "\n".join(f"$ cmd {i} &amp; more" for i in range(300))
            + "</blockquote>\n<i>tail " + "y" * 3000 + "</i>")
    chunks = split_html(text, 1000)
    assert len(chunks) > 5 and all(len(c) <= 1000 for c in chunks)
    for c in chunks:
        opened = re.findall(r"<(b|i|blockquote)[ >]", c)
        closed = re.findall(r"</(b|i|blockquote)>", c)
        assert sorted(opened) == sorted(closed), c[:80]
        assert not re.search(r"&[#\w]*$", c.split("<")[0])  # entities are never cut in half
    assert "".join(strip_html(c) for c in chunks).replace("\n", "") == strip_html(text).replace("\n", "")


def test_fold_long_answers():
    from overseer.core.richtext import fold_long, md_to_html
    long = md_to_html("\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(10)))
    folded = fold_long(long)
    assert folded.startswith("Paragraph 0") and "<blockquote expandable>Paragraph 1" in folded
    with_code = md_to_html("\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(10)) + "\n\n```\nx\n```")
    assert "<blockquote" not in fold_long(with_code)  # a code block can't go inside the fold
    assert fold_long("short") == "short"


QUOTA = """
Disk quotas for ab11578 (uid 4167280):
                             DISK SPACE                # FILES (1000's)
          filesystem       size      quota            number      quota
                      --------------------------   --------------------------
               /home       31GB       50GB ( 63%)       260       500 ( 52%)
            /scratch      678GB     5000GB ( 14%)       301       500 ( 60%)
            /archive        4KB     5120GB (  0%)         0       125 (  0%)
"""


def test_quota_table():
    from overseer.hpc.quota import limits_line, over, parse_quota
    rows = parse_quota(QUOTA)
    assert [(q.fs, q.used, q.limit, q.pct, q.files, q.files_limit, q.files_pct) for q in rows] == [
        ("/home", "31GB", "50GB", 63, 260_000, 500_000, 52),
        ("/scratch", "678GB", "5000GB", 14, 301_000, 500_000, 60),
        ("/archive", "4KB", "5120GB", 0, 0, 125_000, 0),
    ]
    assert limits_line(rows) == ("`/home`: 50GB and 500k files; `/scratch`: 5000GB and 500k files; "
                                 "`/archive`: 5120GB and 125k files")
    assert over(rows, 80) == []
    assert over(rows, 60) == ["/home: 31GB of 50GB (63%)", "/scratch: 301k of 500k files (60%)"]
    assert parse_quota("quota: command not found") == []


def test_job_arrays_fold_into_one_job():
    from overseer.hpc.slurm import Job, _task_count, array_parent, fold_arrays
    assert array_parent("18771337_[2-11%4]") == "18771337" and array_parent("99") == "99"
    assert _task_count("1_[2-11,15,20-21%4]") == 13 and _task_count("1_4") == 1
    (a, b) = fold_arrays([Job("1_1", state="RUNNING", elapsed="5:00"), Job("1_[2-11]", state="PENDING"),
                          Job("2", state="PENDING")])
    assert (a.job_id, a.state, a.elapsed, a.extra["tasks"]) == ("1", "RUNNING", "5:00", "10 PENDING, 1 RUNNING")
    assert b.job_id == "2" and not b.extra
    (c,) = fold_arrays([Job("3_1", state="COMPLETED"), Job("3_2", state="TIMEOUT"), Job("3_3", state="OUT_OF_MEMORY")])
    assert c.state == "OUT_OF_MEMORY" and c.extra["tasks"] == "1 COMPLETED, 1 TIMEOUT, 1 OUT_OF_MEMORY"


# FB1: parameter counts read as metrics; JSON records ignored.
DIAG_LOG = """Model summary:
  encoder: 1,234,567
  reward: 656,129
  critic 98,305 params
  value_head: 4096 parameters
{"phase": "before", "achievement_pct": 3.2, "return_mean": 1.45, "episodes": 20}
step 1000 loss=0.5
eval: {'phase': 'mid', 'achievement_pct': 5.0}
{"phase": "after", "achievement_pct": 7.9, "return_mean": 2.31, "episodes": 20, "nested": {"score": 0.7}}
"""


def test_parameter_counts_are_not_metrics_and_json_records_are():
    from overseer.hpc.diagnose import all_metrics
    m = all_metrics(DIAG_LOG)
    assert "reward" not in m and "critic" not in m and "value_head" not in m, m
    assert m["achievement_pct"] == "7.9" and m["return_mean"] == "2.31", m  # the latest record wins
    assert m["nested/score"] == "0.7" and m["step"] == "1000" and m["loss"] == "0.5"
    assert "phase" not in m  # not a number
    shown = extract_metrics(DIAG_LOG)
    assert "reward" not in shown and shown.get("return_mean") == "2.31", shown


def test_array_task_logs():
    from overseer.hpc.slurm import Job, fold_arrays, log_template, task_states
    assert log_template("/r/pair_42_4294967294.out", "42") == "/r/pair_42_%a.out"
    assert log_template("/r/slurm-42_7.out", "42", "7") == "/r/slurm-42_%a.out"
    assert log_template("/r/run/t3.log", "42", "3") == "/r/run/t%a.log"
    assert log_template("/r/run3/t3.log", "42", "3") == ""  # can't tell which 3 is the task's
    assert log_template("/r/out.log", "42", "3") == ""
    (a,) = fold_arrays([Job("5_0", state="COMPLETED"), Job("5_1", state="RUNNING"), Job("5_[2-3]", state="PENDING")])
    assert a.extra["task"] == "1" and task_states(a) == [("0", "COMPLETED"), ("1", "RUNNING")]
    assert a.extra["task_count"] == "4"


def test_cluster_parsing():
    from overseer.hpc.cluster import fairshare, gpu_count, gpus_allocated, partitions, start_estimates, until
    from overseer.hpc.slurm import parse_scontrol
    assert gpu_count("gpu:a100:4(S:0-1)") == (4, ["a100"]) and gpu_count("(null)") == (0, [])
    assert gpu_count("gpu:a100:2,gpu:v100:2(S:0)") == (4, ["a100", "v100"]) and gpu_count("gpu:2") == (2, [])
    assert gpus_allocated("cpu=8,mem=64G,gres/gpu=3,gres/gpu:a100=3") == 3
    assert gpus_allocated("cpu=8,gres/gpu:a100=1,gres/gpu:v100=2") == 3 and gpus_allocated("") == 0
    nodes = [parse_scontrol(x) for x in [
        "NodeName=g1 CPUAlloc=16 CPUTot=64 Gres=gpu:a100:4(S:0-1) RealMemory=512000 AllocMem=128000 "
        "State=MIXED Partitions=nvidia,preempt AllocTRES=cpu=16,gres/gpu=2",
        "NodeName=g2 CPUAlloc=0 CPUTot=64 Gres=gpu:a100:4 RealMemory=512000 AllocMem=0 State=IDLE+DRAIN "
        "Partitions=nvidia AllocTRES=",
        "NodeName=c1 CPUAlloc=0 CPUTot=128 Gres=(null) RealMemory=256000 AllocMem=0 State=IDLE Partitions=compute "
        "AllocTRES="]]
    parts = {p.name: p for p in partitions(nodes, "compute*|up|2-00:00:00\nnvidia|up|4-00:00:00\n")}
    nv = parts["nvidia"]
    assert (nv.nodes, nv.usable, nv.gpus, nv.gpus_free, nv.max_free_gpus, nv.cpus_free) == (2, 1, 8, 2, 2, 48)
    assert nv.time_limit == "4-00:00:00" and nv.gpu_types == ["a100"] and round(nv.max_free_mem_gb) == 375
    assert parts["compute"].default and parts["compute"].idle == 1 and parts["preempt"].gpus_free == 2
    est = start_estimates("5_[2-3]|2026-10-17T15:00:00|Priority\n5_1|2026-10-17T14:00:00|Resources\n6|N/A|Priority\n")
    assert est == {"5": ("2026-10-17T14:00:00", "Resources"), "6": ("N/A", "Priority")}
    assert fairshare("ab11578_acct|0.421053\n") == [("ab11578_acct", "0.421053")]
    assert until("2026-10-17T14:05:00", "2026-10-17T13:00:00") == "in ~1h05m"
    assert until("N/A", "2026-10-17T13:00:00") == ""


def test_search_query():
    import pytest
    from overseer.core.search import SearchError, fts_query, parse_since
    assert fts_query("ratio 64") == '"ratio" "64"*'
    assert fts_query('"batch size" OR lr') == '"batch size" OR "lr"*'
    assert fts_query('lr=3e-4 NEAR(') == '"lr 3e 4" "NEAR"*'
    assert fts_query('"exact phrase"') == '"exact phrase"'
    with pytest.raises(SearchError):
        fts_query("  -- ")
    assert parse_since("") == 0 and parse_since("7d") > 0
    with pytest.raises(SearchError):
        parse_since("yesterday")


def sign_init_data(token: str, user_id: int, auth_date: int | None = None, extra: str = "") -> str:
    """Telegram's Mini App launch data, signed the way Telegram signs it."""
    import hashlib
    import hmac
    import time
    from urllib.parse import urlencode
    fields = {"auth_date": str(auth_date or int(time.time())), "query_id": "AAH",
              "user": json.dumps({"id": user_id, "first_name": "A"})}
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields) + extra


def test_init_data_verification():
    import time
    from overseer.core.dashboard import verify_init_data
    tok = "123:abc"
    assert verify_init_data(sign_init_data(tok, 42), tok, [42]) == 42
    assert verify_init_data(sign_init_data(tok, 42), tok, [7]) is None  # not an allowed user
    assert verify_init_data(sign_init_data("999:other", 42), tok, [42]) is None  # another bot's
    assert verify_init_data(sign_init_data(tok, 42, int(time.time()) - 2 * 86400), tok, [42]) is None  # too old
    assert verify_init_data(sign_init_data(tok, 42, extra="&user=x"), tok, [42]) is None  # a field twice
    assert verify_init_data(sign_init_data(tok, 42).replace("A%22", "B%22"), tok, [42]) is None  # changed
    assert verify_init_data("", tok, [42]) is None and verify_init_data("garbage", tok, [42]) is None


def test_core_never_imports_a_domain():
    """The platform (overseer/core) knows domains only through core/domain.py's hooks: no core module imports a
    domain's code (overseer/hpc, ...). Domains import core, never the other way round."""
    import ast
    from pathlib import Path
    core = Path(__file__).resolve().parent.parent / "overseer" / "core"
    domains = {p.name for p in core.parent.iterdir() if p.is_dir() and p.name not in ("core", "__pycache__")}
    bad = []
    for f in core.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0] if node.level >= 2 else (mod.split(".")[1] if mod.startswith("overseer.") else "")
                if node.level >= 2 and not mod:  # from .. import hpc
                    top = next((a.name for a in node.names if a.name in domains), "")
                if top in domains:
                    bad.append(f"{f.name}:{node.lineno} from {'.' * node.level}{mod}")
            elif isinstance(node, ast.Import):
                bad += [f"{f.name}:{node.lineno} import {a.name}" for a in node.names
                        if a.name.split(".")[:2] in (["overseer", d] for d in domains)]
    assert domains and not bad, bad
