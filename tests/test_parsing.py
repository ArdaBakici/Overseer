from overseer.diagnose import extract_metrics, extract_traceback, find_issues, tqdm_cleanup
from overseer.notify import split_message
from overseer.slurm import (
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
    from overseer.richtext import md_to_html
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
    from overseer.richtext import split_html, strip_html
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
    from overseer.richtext import fold_long, md_to_html
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
    from overseer.quota import limits_line, over, parse_quota
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
    from overseer.slurm import Job, _task_count, array_parent, fold_arrays
    assert array_parent("18771337_[2-11%4]") == "18771337" and array_parent("99") == "99"
    assert _task_count("1_[2-11,15,20-21%4]") == 13 and _task_count("1_4") == 1
    (a, b) = fold_arrays([Job("1_1", state="RUNNING", elapsed="5:00"), Job("1_[2-11]", state="PENDING"),
                          Job("2", state="PENDING")])
    assert (a.job_id, a.state, a.elapsed, a.extra["tasks"]) == ("1", "RUNNING", "5:00", "10 PENDING, 1 RUNNING")
    assert b.job_id == "2" and not b.extra
    (c,) = fold_arrays([Job("3_1", state="COMPLETED"), Job("3_2", state="TIMEOUT"), Job("3_3", state="OUT_OF_MEMORY")])
    assert c.state == "OUT_OF_MEMORY" and c.extra["tasks"] == "1 COMPLETED, 1 TIMEOUT, 1 OUT_OF_MEMORY"
