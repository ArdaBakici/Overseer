---
name: overseer
description: How to talk to Overseer with the `ov` command - register SLURM jobs you submit, create and manage projects, move or stop tracking jobs, message the researcher, and have yourself woken later (when jobs end, on a timer, or in an approved autopilot loop). Use it whenever you submit a job, when the researcher asks about projects or job tracking, when you want to follow up on something later, or for multi-run experiments.
---

# Overseer and `ov`

Overseer is the system that runs you. It keeps the SSH connection to the HPC, monitors SLURM jobs,
reports on them to the researcher on Telegram, and starts your runs. You only run when the
researcher writes to you, or when something you set up wakes you.

`ov` is a local command that sends requests to Overseer. It answers at once. Every change it makes
is announced to the researcher, so you don't need to ask before using it. It only works inside a run
Overseer started (`$OVERSEER_CTL` holds this run's token).

## Jobs

```bash
ov job add 18746955 18746956     # after EVERY submission: Overseer checks the ids with SLURM, then reports on them
ov job list                      # jobs of this conversation (--all: every job Overseer tracks)
ov job show 18746955             # what Overseer knows: state, folders, logs, final report
ov job move 18746955 dreamer     # the job belongs to project "dreamer" (or "main")
ov job delete 18746955           # stop tracking it. The job keeps running: `scancel` cancels it
```

- Register every job you submit, however you submitted it: sbatch, a script, a Python CLI, an array.
  Overseer also tries to spot submissions on its own, but it can miss some.
- "Delete job X" from the researcher means `ov job delete X`. Cancel it with `scancel` only if they
  also want it stopped.
- Ids are checked: jobs that SLURM doesn't know as the researcher's are refused.

## Projects

```bash
ov project list
ov project new chem --dir /scratch/me/chem --about "DFT with ORCA 6 on the cpu partition" [--move-here]
ov project set chem --dir /scratch/me/chem2 --about "..."
ov project archive chem          # the researcher can bring it back with /project restore chem
ov conversation move chem        # this conversation now belongs to project chem (or "main")
```

- Each project has its own Telegram topic, conversations, jobs and agent settings.
- `--dir` is the project's folder on the HPC. `~/...` refers to the HPC home.
- `--move-here` moves this conversation into the new project. Its folder, description and notes
  apply from your next run.

## Messages and status

```bash
ov notify "epoch 3 done, val loss 0.41 - continuing"   # a message to the researcher right now
ov status                        # this conversation: project, autopilot, jobs, what will wake you
```

Your final reply is sent to the researcher anyway. Use `ov notify` for news in the middle of a
long run.

## Being woken later

Nothing runs between your turns. Don't sleep, poll in a loop or wait for jobs: have Overseer wake
you instead.

```bash
ov watch 18746955 --note "compare with the baseline in runs/base"   # when it ends
ov watch 18746955 18746956                    # when both have ended (or as soon as one fails)
ov wake in 90m "check that eval started"      # at a time
ov watches                                    # what will wake you (ids like w12)
ov unwatch w12                                # or a job id, or "all"
```

When you are woken, the message contains:
- the final report of each finished job (state, exit code, runtime, memory, issues found, the end
  of its log);
- your timer's note;
- which jobs are still running.

### Without autopilot: one-off wake-ups

- Without autopilot, `ov watch` and `ov wake in` wake you once. Use them when you told the
  researcher you'd follow up ("I'll look at the results when it finishes").
- The researcher sees each one and can cancel it.
- Limits:
  - only a few one-offs can be pending at a time;
  - after a few one-off wake-ups in a row without the researcher writing, `ov` refuses more.
    Ask for autopilot then.
- Check-ins while a job runs (`--every`) and repeating timers (`ov wake every`) need autopilot.

## Autopilot: working on your own over many runs

Use autopilot when a task needs several runs, with waiting for results in between: tuning,
ablations, sweeps, "keep going until the score is above X". Don't use it for a single job.

1. **Ask.** Then end your turn.
   ```bash
   ov auto request --goal "crafter seed 1 above score 5 in ~/rf-dreamer; one change per run" \
       --wakeups 20 --hours 48 --checkins 2h
   ```
   The researcher gets an approval card in Telegram. They can start it, decline it, or change the
   number of wake-ups, the time and the check-in interval first. If they approve, you're woken with
   the limits they chose. If they decline, you're told the next time they write.
2. **Start** (the first wake-up):
   - look at the project and earlier results;
   - create `RESEARCH_LOG.md` in the project folder on the HPC (goal, baseline, plan);
   - submit the first experiment, then `ov job add` and `ov watch` it, with `--every 2h` if you want
     check-ins;
   - end your reply.
3. **Each wake-up:**
   - read the results and update the research log (what changed and why, the job ids, the key
     results, the next idea);
   - submit the next experiment and `ov watch` it;
   - end your reply.
   - The wake-up message says which one it is ("wake-up 3 of 20 · 41h left"). The last one tells
     you to wrap up.
4. **Check-ins** (`ov watch <id> --every 2h`):
   - while the job runs you're shown its progress (metrics and the end of its log);
   - if it's healthy, change nothing and reply in a sentence or two;
   - if it's clearly broken, diverging or wasting GPU time, `scancel` it, note why in the log,
     resubmit and watch the new job.
   - Check-ins don't count as wake-ups.
5. **Finish:**
   - `ov auto done "best lr 3e-4: score 5.4 (job 18746990)"` when the goal is reached or nothing
     sensible is left to try;
   - `ov auto help "should I try the larger model? it needs 2x the GPU hours"` when you need the
     researcher's decision. They get a notification and your watches stay.

`ov wake every 6h "summarise the sweep"` sets a repeating timer, and `ov auto status` shows where
you are. Jobs you submit during autopilot but forget to watch are watched for you, as a safety
net. Still: watch them yourself, and group the runs whose results you want to read together.

Rules while on autopilot:
- Change one thing at a time unless the goal says otherwise.
- Give every run its own name and output folder, and never delete or overwrite earlier results.
- Only cancel or change jobs you submitted in this conversation.
- The researcher can stop you any time (`/auto off`, `/stop`). A failed or stopped wake-up pauses
  the loop until they resume it.
