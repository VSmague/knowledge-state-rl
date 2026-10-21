# Learning Self-Representation for Curiosity-driven Deep Reinforcement Learning

Research internship project (Inria, FLOWERS team, Bordeaux — May–October 2026,
supervised by Loris Gaven). Full report: [`report.pdf`](report.pdf).

## Problem

Autotelic agents set their own goals and must decide, among a huge or
unbounded goal space, which ones to practice to learn fastest — uniform
sampling wastes compute on goals already mastered or still hopeless.
Automatic curriculum learning solves this by letting the agent prioritize
goals where its competence is currently improving fastest (*learning
progress*). **MAGELLAN**, a method developed by the team, estimates this
per-goal competence with a network trained continuously through the agent's
own training, which lets it generalize to goals rarely practiced directly.
Its weakness is that the network's prediction target drifts constantly as
the policy itself improves, so two competence estimates taken at different
times are not directly comparable — a progress signal that is fragile in
principle, even though it tends to work well in practice.

This project asks: **can a competence estimator have a prediction target
that stays stable over time, without losing MAGELLAN's ability to
generalize across goals?**

## Approach

The competence estimate is conditioned not on time but on an explicit
**knowledge state** $\phi$: a fixed-size vector summarizing the agent's
recent history of attempted goals and their outcomes, built with a
DeepSet-style encoder (encode each attempt, aggregate, project). Four
aggregation mechanisms are implemented and compared: plain averaging,
learned attention, Transformer self-attention, and an LSTM. On top of
$\phi$, several ways of deriving a learning-progress signal from it are
implemented, including an original *prospective* measure that simulates the
effect of a hypothetical future attempt rather than only comparing past
snapshots.

Three environments, built specifically for this project to stress different
goal-space structures, are used to evaluate it at scale on the Jean Zay
cluster:
- **MiniGrid** — a grid of rooms connected by coloured doors/keys; goal
  difficulty grows with spatial distance from spawn.
- **Crafter** — a Minecraft-like survival/crafting game whose achievements
  form a real technology dependency tree (wood → stone → iron → …).
- **Fetch** — a continuous-goal robotic pick-and-place task whose goal
  count can be scaled arbitrarily without changing per-goal difficulty.

## Results (summary)

- **MiniGrid**: the knowledge-state curriculum (`magellan_ks`) reaches
  80.2% final success vs. 63.8% for MAGELLAN at a 1,599-goal scale, keeps
  its edge at 9,999 goals, and is less prone to the complete-collapse
  failure mode occasionally seen with MAGELLAN.
- **Crafter**: competitive with the best baselines; no method crosses the
  iron-tools tier regardless of training budget — a follow-up with explicit
  reward shaping shows this is a reward-sparsity bottleneck, not a
  curriculum problem.
- **Fetch**: scales robustly to 100,000 simultaneous goals, where the
  simpler ALP-based baseline collapses almost entirely.
- A dedicated set of probes (see the report) examines what $\phi$ actually
  encodes: it tracks true spatial difficulty on MiniGrid, generalizes
  smoothly to unseen nearby goals, stays reliable under noisy histories,
  and — on Fetch — generalizes to goals never practiced during training,
  with a mild cautious (not overconfident) bias.

## Repo structure

```
core/
  train.py          Main PPO + curriculum training loop (non-recurrent agents)
  train_rnn.py       Recurrent (LSTM) variant, crafter_game-only
  curriculum.py       Curriculum strategies: uniform, online, magellan,
                      magellan_ks (DeepSet knowledge-state encoder + ALP)
  eval.py             Evaluation API (success rates per goal/group)
  eval_parallel.py     Multi-process eval pool, used internally by eval.py
  parallel_envs.py     Multi-process vectorized envs for training
  plots.py            Game-agnostic plots (success curve, losses, ks diagnostics)

minigrid_game/
  env.py             Grid-of-rooms env (same Env/Agent contract as the others)
  architecture.py     Agent (CNN encoder + policy/value heads)
  preview_env.py       Static layout snapshot (non-interactive, saves to PNG)

crafter_game/
  env.py             Crafter wrapper (goal-conditioned achievements)
  architecture.py     Agent
  tech_tree.py         Structured achievement-dependency goal features
  goal_llm_embed.py    Precomputes Qwen text embeddings for goal representation
  visualize.py         Interactive viewer (keyboard-controlled)

fetch_game/
  env.py             Continuous pick-and-place goal wrapper (FetchPickAndPlace)
  architecture.py     Agent

tests/               Reproducibility check for core/eval_parallel.py's
                      multi-process eval pool against a single-process
                      reference implementation

requirements.txt     pip dependencies, grouped by which file needs each one
report.pdf           Full internship report
```

Every game module exposes the same interface, so `core/` never branches on
which game is running:

```python
game.env.Env(size=..., seed=...).reset()/.step(action)/.set_goal(goal)
game.env.goals(size) -> list
game.architecture.build_agent(size, device=...) -> Agent
```

`crafter_game/` has extra files (`tech_tree.py`, `goal_llm_embed.py`,
`visualize.py`) because it supports several goal-representation schemes
and is the one env with a full interactive viewer — not because the
other two are incomplete; each module only carries what its own game
genuinely needs.

## Quickstart

```bash
pip install -r requirements.txt

python -m core.train --game minigrid_game --curriculum magellan_ks \
    --size 5 5 8 8 --steps 20_000_000 --n_envs 8 --run_name test1

python -m core.train --game crafter_game --curriculum magellan_ks \
    --steps 5_000_000 --n_envs 8 --run_name test1

python -m core.train --game fetch_game --curriculum magellan_ks \
    --size 1000 --steps 10_000_000 --n_envs 8 --run_name test1

python -m core.plots --run_dir exp/<run>/
```

Curricula (`core/curriculum.py`): `uniform` (random), `online` (per-goal ALP
from raw outcomes), `magellan` (learned competence MLP, time-snapshot ALP),
`magellan_ks` (competence MLP conditioned on the knowledge state $\phi$,
ALP between `C(g | ks_now)` and `C(g | ks_old)` — no weight snapshots
needed, see the class docstring for the stationarity argument).
