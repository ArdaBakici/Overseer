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
