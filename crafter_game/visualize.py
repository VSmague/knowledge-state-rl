"""
crafter_game/visualize.py — Interactive step-by-step visualization of a
trained Crafter policy. Controls:

  [Step]   SPACE : one policy step
  [Reset]  R     : reset episode (new goal if unlocked)
  [Auto]   A     : toggle auto-stepping
  [G]            : lock / unlock goal
  [P]            : greedy (argmax) vs sample
  [F]            : show / hide FOV dimming on the wide view
  Click a goal name in the goal list to set it (when unlocked)

Layout:
  - Left  : wider local world view around the agent; tiles outside the
            agent's true FOV (9×7 terrain) are greyed out when FOV is on
  - Mid   : action probabilities + value
  - Right : raw 64×64 observation the network actually sees
  - Bottom: inventory strip + control buttons + clickable goal list

Usage:
  python -m crafter_game.visualize --ckpt exp/.../ckpts/ckpt_pct100.pt \
      --goal collect_wood
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from crafter.engine import LocalView

from crafter_game.env import (
    Env, ACHIEVEMENTS, N_ACTIONS, goals as crafter_goals,
    resolve_goal_repr as resolve_crafter_goal_repr,
)
from crafter_game.architecture import build_agent


BG_DARK = "#1A1A2E"
TEXT_C  = "#A0A8C8"
PANEL_C = "#0F1923"

# Short labels that fit the bar chart; full names stay in tooltips / status.
ACTION_NAMES = [
    "noop", "←", "→", "↑", "↓", "do", "sleep",
    "stone", "table", "furnace", "plant",
    "w.pick", "s.pick", "i.pick",
    "w.sword", "s.sword", "i.sword",
]


def _render_wide_view(inner_env, fov_grid=(9, 7), wide_grid=(21, 21),
                      unit=8, dim_outside=True):
    """
    Render a wide LocalView centered on the player. If dim_outside, tiles
    outside the agent's terrain FOV (fov_grid, matching crafter's default
    local-view size) are multiplied by a grey veil.
    """
    player = inner_env._player
    textures = inner_env._textures
    world = inner_env._world

    wide = LocalView(world, textures, list(wide_grid))
    canvas = wide(player, np.array([unit, unit])).astype(np.float32)

    if not dim_outside:
        return np.clip(canvas, 0, 255).astype(np.uint8)

    # LocalView uses offset = grid // 2; FOV is the central fov_grid tiles.
    wx, wy = wide_grid
    fx, fy = fov_grid
    ox, oy = wx // 2, wy // 2
    half_fx, half_fy = fx // 2, fy // 2

    mask = np.ones((wx, wy), dtype=np.float32) * 0.35  # outside = dimmed
    x0 = max(0, ox - half_fx)
    x1 = min(wx, ox - half_fx + fx)
    y0 = max(0, oy - half_fy)
    y1 = min(wy, oy - half_fy + fy)
    mask[x0:x1, y0:y1] = 1.0

    # Expand tile mask to pixels (canvas is (W*unit, H*unit, 3) in LocalView
    # convention: first dim = x tiles, second = y tiles).
    pix = np.repeat(np.repeat(mask, unit, axis=0), unit, axis=1)[..., None]
    # Soft grey outside FOV (keep a bit of context, not pure black).
    grey = np.array([40.0, 42.0, 55.0], dtype=np.float32)
    out = canvas * pix + grey * (1.0 - pix)
    return np.clip(out, 0, 255).astype(np.uint8)


class Visualizer:
    def __init__(self, agent, env, valid_goals, device,
                 fixed_goal=None, greedy=True, action_names=None):
        self.agent       = agent
        self.env         = env
        self.valid_goals = list(valid_goals)
        self.device      = device
        self.fixed_goal  = fixed_goal
        self.greedy      = greedy
        self.action_names = action_names or ACTION_NAMES

        self.step_count   = 0
        self.total_reward = 0.0
        self.last_action  = None
        self.done         = False
        self.success      = False
        self.auto         = False
        self.show_fov     = True
        self.last_info    = {}

        self.goal        = fixed_goal or self._sample_goal()
        self.goal_locked = fixed_goal is not None
        self.obs         = self._reset()

        self.fig = plt.figure(figsize=(15.5, 8.5), facecolor=BG_DARK)

        # Main panels
        self.ax_world = self.fig.add_axes([0.02, 0.28, 0.42, 0.68])
        self.ax_prob  = self.fig.add_axes([0.47, 0.28, 0.22, 0.68])
        self.ax_obs   = self.fig.add_axes([0.72, 0.48, 0.26, 0.48])
        self.ax_inv   = self.fig.add_axes([0.72, 0.28, 0.26, 0.16])
        self.ax_goals = self.fig.add_axes([0.02, 0.12, 0.96, 0.12])

        for ax in (self.ax_world, self.ax_prob, self.ax_obs,
                   self.ax_inv, self.ax_goals):
            ax.set_facecolor(PANEL_C)

        # Buttons
        btn_style = dict(color="#1D3557", hovercolor="#2C5282")
        ax_step  = self.fig.add_axes([0.02, 0.02, 0.10, 0.07])
        ax_reset = self.fig.add_axes([0.13, 0.02, 0.10, 0.07])
        ax_auto  = self.fig.add_axes([0.24, 0.02, 0.10, 0.07])
        ax_unlk  = self.fig.add_axes([0.35, 0.02, 0.12, 0.07])
        ax_grdy  = self.fig.add_axes([0.48, 0.02, 0.12, 0.07])
        ax_fov   = self.fig.add_axes([0.61, 0.02, 0.12, 0.07])

        self.btn_step  = Button(ax_step,  "Step  [SPACE]", **btn_style)
        self.btn_reset = Button(ax_reset, "Reset  [R]",    **btn_style)
        self.btn_auto  = Button(ax_auto,  "Auto  [A]",     **btn_style)
        self.btn_unlk  = Button(
            ax_unlk,
            "Goal: LOCKED [G]" if self.goal_locked else "Goal: free [G]",
            color="#7A2E2E" if self.goal_locked else "#1D3557",
            hovercolor="#9A3E3E" if self.goal_locked else "#2C5282")
        self.btn_grdy  = Button(
            ax_grdy,
            "Mode: Greedy [P]" if self.greedy else "Mode: Sample [P]",
            **btn_style)
        self.btn_fov = Button(
            ax_fov,
            "FOV: ON [F]" if self.show_fov else "FOV: OFF [F]",
            color="#2DC653" if self.show_fov else "#1D3557",
            hovercolor="#2C5282")

        for btn in (self.btn_step, self.btn_reset, self.btn_auto,
                    self.btn_unlk, self.btn_grdy, self.btn_fov):
            btn.label.set_color("white")
            btn.label.set_fontsize(8)

        self.btn_step.on_clicked(lambda _: self._step())
        self.btn_reset.on_clicked(lambda _: self._reset_cb())
        self.btn_auto.on_clicked(lambda _: self._toggle_auto())
        self.btn_unlk.on_clicked(lambda _: self._toggle_goal_lock())
        self.btn_grdy.on_clicked(lambda _: self._toggle_greedy())
        self.btn_fov.on_clicked(lambda _: self._toggle_fov())

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._timer = self.fig.canvas.new_timer(interval=250)
        self._timer.add_callback(self._auto_tick)

        self._draw()
        plt.show()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _sample_goal(self):
        return self.valid_goals[np.random.randint(len(self.valid_goals))]

    def _reset(self):
        self.step_count   = 0
        self.total_reward = 0.0
        self.last_action  = None
        self.done         = False
        self.success      = False
        self.last_info    = {}
        self.env.set_goal(self.goal)
        obs, _ = self.env.reset()
        return obs

    def _reset_cb(self):
        if not self.goal_locked:
            self.goal = self._sample_goal()
        self.obs = self._reset()
        self._draw()

    def _get_action_probs(self):
        obs_t = torch.from_numpy(self.obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lat = self.agent.encoder(obs_t)
            gv = self.agent.goal_repr([self.goal])
            logits, val = self.agent.ac(lat, gv)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(val.item())

    def _step(self):
        if self.done:
            self._reset_cb()
            return

        probs, _ = self._get_action_probs()
        if self.greedy:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        self.last_action = action

        obs, reward, term, trunc, info = self.env.step(action)
        self.obs = obs
        self.last_info = info
        self.total_reward += reward
        self.step_count   += 1
        self.done    = term or trunc
        self.success = bool(term)
        self._draw()

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw(self):
        probs, value = self._get_action_probs()
        self._draw_world()
        self._draw_probs(probs, value)
        self._draw_obs()
        self._draw_inventory()
        self._draw_goals()
        self.fig.canvas.draw_idle()

    def _draw_world(self):
        self.ax_world.clear()
        self.ax_world.set_facecolor(PANEL_C)
        self.ax_world.set_xticks([]); self.ax_world.set_yticks([])

        inner = self.env._env
        # Crafter LocalView returns (x, y, 3); imshow wants (row=y, col=x).
        canvas = _render_wide_view(
            inner, fov_grid=(9, 7), wide_grid=(21, 21), unit=8,
            dim_outside=self.show_fov)
        img = np.transpose(canvas, (1, 0, 2))
        self.ax_world.imshow(img, interpolation="nearest")

        # FOV rectangle outline (in pixel coords of the transposed image).
        unit, wx, wy = 8, 21, 21
        fx, fy = 9, 7
        ox, oy = wx // 2, wy // 2
        # After transpose: row = y tile, col = x tile
        col0 = (ox - fx // 2) * unit - 0.5
        row0 = (oy - fy // 2) * unit - 0.5
        if self.show_fov:
            self.ax_world.add_patch(mpatches.Rectangle(
                (col0, row0), fx * unit, fy * unit,
                fill=False, edgecolor="#FFD700", linewidth=1.5, zorder=5))

        status = (f"Step {self.step_count}  |  reward {self.total_reward:.1f}"
                  f"  |  goal: {self.goal}")
        if self.last_action is not None:
            name = self.action_names[self.last_action]
            status += f"  |  last: {name}"
        if self.done and self.success:
            status += "  [SUCCESS]"
            color = "#2DC653"
        elif self.done:
            status += "  [DEAD/TIMEOUT]"
            color = "#E63946"
        else:
            color = "white"
        self.ax_world.set_title(status, color=color, fontsize=9, pad=6)
        self.ax_world.set_xlabel(
            "Wide view (21×21) — yellow box = agent FOV (9×7)"
            if self.show_fov else "Wide view (21×21) — FOV dimming OFF",
            color=TEXT_C, fontsize=8)

    def _draw_probs(self, probs, value):
        self.ax_prob.clear()
        self.ax_prob.set_facecolor(PANEL_C)
        colors = ["#E94560" if i == int(np.argmax(probs)) else "#457B9D"
                  for i in range(N_ACTIONS)]
        # Reverse so noop is at the top.
        y = np.arange(N_ACTIONS)
        self.ax_prob.barh(y, probs, color=colors, alpha=0.85)
        self.ax_prob.set_yticks(y)
        labels = [self.action_names[i] if i < len(self.action_names)
                  else ACTION_NAMES[i] for i in range(N_ACTIONS)]
        self.ax_prob.set_yticklabels(labels, color="white", fontsize=8)
        self.ax_prob.invert_yaxis()
        self.ax_prob.set_xlim(0, 1)
        self.ax_prob.set_xlabel("Probability", color=TEXT_C, fontsize=8)
        self.ax_prob.tick_params(colors=TEXT_C)
        for spine in self.ax_prob.spines.values():
            spine.set_edgecolor("#333355")
        for i, p in enumerate(probs):
            if p >= 0.02:
                self.ax_prob.text(p + 0.02, i, f"{p:.2f}",
                                  va="center", color="white", fontsize=7)
        self.ax_prob.set_title(f"Action probs  (V={value:.3f})",
                               color=TEXT_C, fontsize=9, pad=4)

    def _draw_obs(self):
        self.ax_obs.clear()
        self.ax_obs.set_facecolor(PANEL_C)
        self.ax_obs.set_xticks([]); self.ax_obs.set_yticks([])
        self.ax_obs.imshow(self.obs, interpolation="nearest")
        self.ax_obs.set_title("Agent obs (64×64) — network input",
                              color=TEXT_C, fontsize=8, pad=4)

    def _draw_inventory(self):
        self.ax_inv.clear()
        self.ax_inv.set_facecolor(PANEL_C)
        self.ax_inv.set_xticks([]); self.ax_inv.set_yticks([])
        for spine in self.ax_inv.spines.values():
            spine.set_edgecolor("#333355")

        inv = {}
        health = "?"
        try:
            player = self.env._env._player
            inv = {k: v for k, v in player.inventory.items() if v}
            health = player.health
        except Exception:
            pass

        lines = [f"health: {health}"]
        if inv:
            items = [f"{k}:{v}" for k, v in inv.items()]
            # Wrap into ~2 lines.
            mid = (len(items) + 1) // 2
            lines.append("  ".join(items[:mid]))
            if mid < len(items):
                lines.append("  ".join(items[mid:]))
        else:
            lines.append("(empty inventory)")

        text = "\n".join(lines)
        self.ax_inv.text(0.02, 0.95, text, transform=self.ax_inv.transAxes,
                         va="top", ha="left", color="white", fontsize=8,
                         family="monospace")
        self.ax_inv.set_title("Inventory", color=TEXT_C, fontsize=8, pad=2)

    def _draw_goals(self):
        self.ax_goals.clear()
        self.ax_goals.set_facecolor(PANEL_C)
        self.ax_goals.set_xticks([]); self.ax_goals.set_yticks([])
        for spine in self.ax_goals.spines.values():
            spine.set_edgecolor("#333355")

        n = len(self.valid_goals)
        # 2 rows × 11 columns for 22 achievements.
        cols = 11
        rows = int(np.ceil(n / cols))
        self._goal_hitboxes = []  # list of (goal, (x0,x1,y0,y1)) in axes frac

        for i, g in enumerate(self.valid_goals):
            r, c = divmod(i, cols)
            x = (c + 0.5) / cols
            y = 1.0 - (r + 0.5) / rows
            is_cur = (g == self.goal)
            color = "#FFD700" if is_cur else TEXT_C
            weight = "bold" if is_cur else "normal"
            self.ax_goals.text(x, y, g, transform=self.ax_goals.transAxes,
                               ha="center", va="center", color=color,
                               fontsize=7, fontweight=weight)
            # Approximate hitbox in axes coordinates.
            self._goal_hitboxes.append((
                g,
                ((c) / cols, (c + 1) / cols,
                 1.0 - (r + 1) / rows, 1.0 - r / rows),
            ))

        lock = "LOCKED" if self.goal_locked else "click to set"
        self.ax_goals.set_title(
            f"Goals ({lock}) — current: {self.goal}",
            color=TEXT_C, fontsize=8, pad=2)

    # ── Controls ─────────────────────────────────────────────────────────

    def _toggle_auto(self):
        self.auto = not self.auto
        self.btn_auto.label.set_text("Auto ON [A]" if self.auto else "Auto  [A]")
        self.btn_auto.color = "#2DC653" if self.auto else "#1D3557"
        self.btn_auto.ax.set_facecolor(self.btn_auto.color)
        if self.auto:
            self._timer.start()
        else:
            self._timer.stop()

    def _toggle_goal_lock(self):
        self.goal_locked = not self.goal_locked
        self.btn_unlk.label.set_text(
            "Goal: LOCKED [G]" if self.goal_locked else "Goal: free [G]")
        self.btn_unlk.color = "#7A2E2E" if self.goal_locked else "#1D3557"
        self.btn_unlk.ax.set_facecolor(self.btn_unlk.color)
        self._draw_goals()
        self.fig.canvas.draw_idle()

    def _toggle_greedy(self):
        self.greedy = not self.greedy
        self.btn_grdy.label.set_text(
            "Mode: Greedy [P]" if self.greedy else "Mode: Sample [P]")
        self.fig.canvas.draw_idle()

    def _toggle_fov(self):
        self.show_fov = not self.show_fov
        self.btn_fov.label.set_text("FOV: ON [F]" if self.show_fov else "FOV: OFF [F]")
        self.btn_fov.color = "#2DC653" if self.show_fov else "#1D3557"
        self.btn_fov.ax.set_facecolor(self.btn_fov.color)
        self._draw()

    def _auto_tick(self):
        if not self.auto:
            return
        self._step()
        if self.done:
            # Pause briefly on terminal, then continue if still auto.
            pass

    def _on_click(self, event):
        if event.inaxes != self.ax_goals:
            return
        if self.goal_locked:
            print("Goal is locked — press [G] to unlock before picking another.")
            return
        if event.xdata is None or event.ydata is None:
            return
        # Convert display click to axes fraction.
        bbox = self.ax_goals.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        xf = (event.x - bbox.x0) / bbox.width
        yf = (event.y - bbox.y0) / bbox.height
        for g, (x0, x1, y0, y1) in self._goal_hitboxes:
            if x0 <= xf <= x1 and y0 <= yf <= y1:
                self.goal = g
                self.obs = self._reset()
                self._draw()
                print(f"Goal set to {self.goal}")
                return

    def _on_key(self, event):
        if event.key == " ":
            self._step()
        elif event.key == "r":
            self._reset_cb()
        elif event.key == "a":
            self._toggle_auto()
        elif event.key == "g":
            self._toggle_goal_lock()
        elif event.key == "p":
            self._toggle_greedy()
        elif event.key == "f":
            self._toggle_fov()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive Crafter policy visualization")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--goal", default=None,
                   help="Fix goal to this achievement name "
                        "(e.g. collect_wood). If omitted, sampled randomly.")
    p.add_argument("--size", type=int, nargs=2, default=None,
                   metavar=("W", "H"),
                   help="World size in tiles (default: from ckpt, else 64 64)")
    p.add_argument("--length", type=int, default=None,
                   help="Max episode steps (default: env default 10000)")
    p.add_argument("--enc_dim", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--crafter_goal_repr",
                   choices=["onehot", "tech_tree", "llm"],
                   default="onehot",
                   help="Must match training: onehot|tech_tree|llm")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample", action="store_true",
                   help="Start in stochastic sampling mode (toggle with [P])")
    return p.parse_args()


def main():
    matplotlib.use("TkAgg")
    args = parse_args()
    device = args.device

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    size = tuple(args.size) if args.size else tuple(ckpt.get("size", (64, 64)))
    enc_dim = args.enc_dim if args.enc_dim is not None else ckpt.get("enc_dim", 512)
    hidden_dim = (args.hidden_dim if args.hidden_dim is not None
                  else ckpt.get("hidden_dim", 1024))

    agent_kwargs = dict(enc_dim=enc_dim, hidden_dim=hidden_dim)
    emb_matrix, emb_path = resolve_crafter_goal_repr(
        args.crafter_goal_repr, list(ACHIEVEMENTS))
    if emb_matrix is not None:
        # GoalRepr loads from .npy — materialize next to ckpt if possible.
        import os
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        emb_path = os.path.join(ckpt_dir, "tech_tree_goal_embeddings.npy")
        if not os.path.isfile(emb_path):
            np.save(emb_path, emb_matrix)
    if emb_path:
        agent_kwargs["frozen_embeddings_path"] = emb_path

    agent = build_agent(size, device=device, **agent_kwargs)
    agent.encoder.load_state_dict(ckpt["encoder"])
    agent.goal_repr.load_state_dict(ckpt["goal_repr"])
    agent.ac.load_state_dict(ckpt["ac"])
    agent.encoder.eval(); agent.goal_repr.eval(); agent.ac.eval()

    env = Env(size=size, length=args.length)
    valid_goals = crafter_goals(size)
    action_names = list(env._env.action_names)

    fixed_goal = args.goal
    if fixed_goal is not None and fixed_goal not in valid_goals:
        print(f"Warning: goal {fixed_goal!r} not in achievements, ignoring.")
        fixed_goal = None

    print(f"Loaded ckpt: {args.ckpt}")
    print(f"  game={ckpt.get('game')} curriculum={ckpt.get('curriculum')} "
          f"step={ckpt.get('step')}")
    print(f"  size={size} enc_dim={enc_dim} hidden_dim={hidden_dim}")
    print("Controls: SPACE=step  R=reset  A=auto  G=lock goal  "
          "P=greedy/sample  F=FOV  click goal name=set goal")

    Visualizer(agent, env, valid_goals, device, fixed_goal,
               greedy=not args.sample, action_names=action_names)


if __name__ == "__main__":
    main()
