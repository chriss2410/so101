# Tomorrow's Cheatsheet (Day 2)

This is state as of **2026-07-07 evening**. Read this first tomorrow — it's the shortest path back to a working inference loop.

## Where we left off

- ✅ SO-101 arms calibrated (follower + leader), motors flashed, ports known
- ✅ Dataset `chris241094/d-com-0_20260707_083022` on HF (10 episodes, 5379 frames)
- ✅ ACT policy trained: `chris241094/act-d-com-0` (10k steps — undertrained but functional)
- ✅ AWS `research-1xA10` provisioned, `serve.sh` installed, port 7860 open on SAP VPN
- ✅ Remote inference **worked end-to-end** — arm moved, camera streamed, policy inferred on cuda
- ⚠️ Leader arm had a "Missing motor IDs" issue during teleop earlier — likely a loose 3-pin cable on the leader. This does NOT block `infer-remote` (which only uses the follower).

## Instance state at end of day 2026-07-07

Run this first thing tomorrow to confirm:

```bash
export PATH="$HOME/.local/bin:$PATH"
ecm ec2 info research-1xA10 | grep -E "State|Public IP"
```

- **If `State = stopped`**: you correctly shut it down at end of day. Do `ecm ec2 start research-1xA10`, wait 30-60s, then check the Public IP — **it may have changed**. If it did, update `SERVER_ADDRESS` in `so101/.env`.
- **If `State = running`**: you forgot to stop it (cost you ~$1.20/hr overnight but no drama).

## The 4-command warmup

Assuming instance is up and Public IP hasn't changed from `52.59.241.221`:

```bash
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 serve start                    # ~2s, brings up policy server
uv run so101 serve status                   # confirm RUNNING, port bound
uv run so101 ui                             # gradio dashboard (or `so101 infer-remote` for the bare CLI)
```

If the exec line in `so101 infer-remote` output shows a `.venv/bin/python` path that is NOT `/Users/i539735/dev/physical-ai/research/so101/.venv/bin/python`, see [DEPLOYMENT.md § Troubleshooting](DEPLOYMENT.md) — wrong-venv issue.

## Terminal setup for debugging

If anything looks flaky, use two terminals:

**Terminal 1** — server-side logs (this is where crashes will show tracebacks):
```bash
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 serve logs -f
```

**Terminal 2** — inference:
```bash
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 infer-remote
```

## Yesterday's discovered gotchas (avoid these tomorrow)

1. **Don't activate the ec2-manager venv, then run `so101` from another dir**. It resolves the wrong Python. Fixed in code but you can still trip on it if `so101` gets accidentally installed into another venv. Always `cd so101/` first, always `deactivate 2>/dev/null` if a `(name)` prompt is showing.

2. **Public IP changes on stop/start**. Update `.env` if it changes.

3. **Leader arm's 3-pin motor cable is easy to knock loose**. If teleop errors with "Missing motor IDs", check the cable seating physically. But you don't need the leader for `infer-remote`.

4. **A stale Rerun window blocks new ones**. Silent failure — the second session either connects to the invisible old window or doesn't open at all. Fix: `pkill -f rerun` and re-run.

5. **`manual` recording mode had a bug where reset time was 0** (fixed 2026-07-07). Both `episode_time_s` and `reset_time_s` are now 24h in manual mode. Keyboard is the only pacing signal.

## Model expectations (managing expectations for the eval)

- Trained on 10 episodes, 10k steps, single task.
- **Will move smoothly and continuously** if the pipeline is working.
- **Will NOT reliably complete the pick-and-stack task**. Undertrained. Expect it to approach the cup roughly but miss on gripping.
- If you want a policy that actually works, either:
  - **Record more data** (30+ episodes) and retrain longer (50k+ steps), or
  - **Rerun training** on the existing dataset with 100k steps (~4h on A10G)

## When you're done for the day

```bash
uv run so101 serve stop                     # frees GPU memory on the server
ecm ec2 stop research-1xA10                 # stops paying $1.20/hr
```

Do both, in this order. `ecm ec2 stop` alone would let the server process die when the box shuts down but leaves the tmux artifacts around; explicit `serve stop` is cleaner.

## If you decide to retrain with more data

Full recording session (manual pacing):
```bash
uv run so101 record --manual --auto-name       # keyboard-driven, next d-com-N
```

Longer training on AWS (100k steps on the same dataset takes ~4h on A10G):
```bash
ssh research-1xA10 -t 'tmux new-session -A -s train "source /opt/dlami/nvme/train-so101/env.sh && lerobot-train --dataset.repo_id=chris241094/d-com-0_20260707_083022 --policy.type=act --policy.device=cuda --policy.push_to_hub=true --policy.repo_id=chris241094/act-d-com-0-v2 --output_dir=/opt/dlami/nvme/outputs/act-d-com-0-v2 --job_name=act_d-com-0_v2 --batch_size=16 --steps=100000 --save_freq=10000 --wandb.enable=true"'
# Ctrl-b d to detach. Reconnect: ssh research-1xA10 -t 'tmux attach -t train'
```

When it finishes, update `POLICY_PATH=chris241094/act-d-com-0-v2` in `.env` and re-run `so101 infer-remote`.

## Full docs

- **This file** (day 2 cheatsheet) — fast recovery of yesterday's state
- [DEPLOYMENT.md](DEPLOYMENT.md) — thorough deployment runbook + troubleshooting
- [WORKFLOW.md](WORKFLOW.md) — the whole pipeline (teleop → record → train → deploy)
- [../README.md](../README.md) — project overview
