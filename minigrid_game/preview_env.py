"""
minigrid_game/preview_env.py — Render a static image of a full
minigrid_game env layout (all rooms, walls, keys, doors, goal, agent)
for visual inspection, non-interactive (matplotlib "Agg" backend, saves
to a file).

Usage:
    python -m minigrid_game.preview_env --size 5 5 5 5
    python -m minigrid_game.preview_env --size 5 5 5 5 --seed 42
    python -m minigrid_game.preview_env --size 5 5 5 5 --seed 42 --out my_preview.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from minigrid_game.env import Env, grid_dims


BG_DARK  = "#1A1A2E"
BG_CELL  = "#16213E"
WALL_C   = "#2C2C54"
TEXT_C   = "#A0A8C8"
ACTION_NAMES = ["←", "→", "↑", "Pickup", "Drop", "Toggle"]

KEY_COLORS_HEX = {
    "yellow": "#F5C518", "red":    "#E63946", "blue":   "#457B9D",
    "purple": "#9B5DE5", "green":  "#2DC653", "grey":   "#8D99AE",
}


def draw_env(ax, mg_env, goal, size, step=0, total_reward=0.0,
            last_action=None, done=False, success=False,
            visible_cells=None, show_room_labels=True):
    """Renders the full 2D room-grid maze layout (walls, doors+colors,
    keys+colors, goal, agent) onto a matplotlib axis, using the standard
    Wall/Door/Key/Goal MiniGrid object types (rooms are still just cells
    on one big Grid). Each room gets its own (col,row) label.
    mg_env is the INNER MiniGridEnv (env._env, not the Env wrapper)."""
    from minigrid.core.world_object import Wall, Key, Door, Goal as GoalObj

    W, H = mg_env.width, mg_env.height
    n_cols, n_rows, w, h = size

    ax.clear()
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    for x in range(W):
        for y in range(H):
            cell = mg_env.grid.get(x, y)

            if isinstance(cell, Wall):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x-.48, y-.48), .96, .96,
                    boxstyle="square,pad=0", lw=0,
                    facecolor=WALL_C, zorder=1))

            elif isinstance(cell, Door):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x-.48, y-.48), .96, .96,
                    boxstyle="square,pad=0", lw=0,
                    facecolor=WALL_C, zorder=1))
                dc = KEY_COLORS_HEX.get(cell.color, "#aaa")
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x-.28, y-.38), .56, .76,
                    boxstyle="round,pad=0.04", lw=1.5,
                    edgecolor=dc, facecolor=BG_DARK, zorder=3))
                ax.text(x, y, "X" if cell.is_locked else "O",
                        ha="center", va="center", fontsize=6,
                        color=dc, fontweight="bold", zorder=4)

            elif isinstance(cell, Key):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x-.46, y-.46), .92, .92,
                    boxstyle="square,pad=0", lw=0,
                    facecolor=BG_CELL, zorder=1))
                ax.text(x, y, "K",
                        ha="center", va="center", fontsize=8,
                        color=KEY_COLORS_HEX.get(cell.color, "#aaa"),
                        fontweight="bold", zorder=4)

            elif isinstance(cell, GoalObj):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x-.46, y-.46), .92, .92,
                    boxstyle="square,pad=0", lw=0,
                    facecolor="#0F3460", zorder=1))
                ax.add_patch(mpatches.RegularPolygon(
                    (x, y), numVertices=5, radius=.26,
                    orientation=np.pi/2, color="#FFD700", zorder=3))

            else:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x-.46, y-.46), .92, .92,
                    boxstyle="square,pad=0", lw=0,
                    facecolor=BG_CELL, zorder=1))

    gx, gy = goal
    cell = mg_env.grid.get(gx, gy)
    if not isinstance(cell, GoalObj):
        ax.add_patch(mpatches.FancyBboxPatch(
            (gx-.46, gy-.46), .92, .92,
            boxstyle="square,pad=0", lw=2,
            edgecolor="#FFD700", facecolor=BG_CELL, zorder=2))

    if visible_cells is not None:
        for x in range(W):
            for y in range(H):
                if (x, y) not in visible_cells:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (x-.5, y-.5), 1.0, 1.0,
                        boxstyle="square,pad=0", lw=0,
                        facecolor=BG_DARK, alpha=0.72, zorder=4.5))

    ax_x, ay = mg_env.agent_pos
    ax.add_patch(plt.Circle((ax_x, ay), .24, color="#E94560", zorder=5))
    dx = [1, 0, -1, 0][mg_env.agent_dir]
    dy = [0, 1,  0, -1][mg_env.agent_dir]
    ax.annotate("",
                xy=(ax_x + dx*.20, ay + dy*.20), xytext=(ax_x, ay),
                arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.12",
                                color="white", lw=1.3), zorder=6)

    if mg_env.carrying:
        kc = KEY_COLORS_HEX.get(mg_env.carrying.color, "#aaa")
        ax.add_patch(plt.Circle((ax_x + .32, ay - .32), .12,
                                color=kc, zorder=7))

    if show_room_labels:
        for r in range(n_rows):
            for c in range(n_cols):
                # Any interior cell maps back to this room via room_id —
                # use the room's own top-left interior corner (matches
                # env.py's _room_x0/_room_y0) as the label anchor.
                x0 = c * (w + 1) + 1
                y0 = (n_rows - 1 - r) * (h + 1) + 1
                ax.text(x0 + w/2 - 0.5, y0 - 0.35, f"({c},{r})",
                        ha="center", va="bottom", color=TEXT_C,
                        fontsize=6, fontweight="bold", zorder=7)

    status = f"Step {step}  |  reward {total_reward:.1f}  |  goal {goal}"
    if last_action is not None:
        status += f"  |  last: {ACTION_NAMES[last_action]}"
    suffix = "  [SUCCESS]" if (done and success) else ("  [TIMEOUT]" if done else "")
    color  = "#2DC653" if (done and success) else ("#E63946" if done else "white")
    ax.set_title(status + suffix, color=color, fontsize=9, pad=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, nargs=4, default=(5, 5, 5, 5),
                   metavar=("N_COLS", "N_ROWS", "ROOM_W", "ROOM_H"))
    p.add_argument("--seed", type=int, default=None,
                   help="None -> a fresh random layout")
    p.add_argument("--goal", type=int, nargs=2, default=None)
    p.add_argument("--out", default="minigrid_game_preview.png")
    args = p.parse_args()

    size = tuple(args.size)
    goal = tuple(args.goal) if args.goal else None

    env = Env(size=size, goal=goal, seed=args.seed)
    obs, info = env.reset()

    W, H = grid_dims(size)

    from minigrid.core.world_object import Key, Door, Goal as GoalObj
    print(f"size={size}  grid={W}x{H}  seed={env._seed}  goal={env.goal}  "
         f"length={env.length}")
    for x in range(W):
        for y in range(H):
            cell = env._env.grid.get(x, y)
            if isinstance(cell, Key):
                print(f"  KEY  at ({x:>2},{y:>2}) color={cell.color}")
            elif isinstance(cell, Door):
                print(f"  DOOR at ({x:>2},{y:>2}) color={cell.color} "
                     f"locked={cell.is_locked}")
            elif isinstance(cell, GoalObj):
                print(f"  GOAL at ({x:>2},{y:>2})")
    print(f"  AGENT spawn: {tuple(env._env.agent_pos)}")

    n_cols, n_rows, w, h = size
    fig, ax = plt.subplots(figsize=(max(10, W * 0.55), max(5, H * 0.55)))
    draw_env(ax, env._env, env.goal, size, step=0, total_reward=0.0)
    plt.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved -> {args.out}")

    env.close()


if __name__ == "__main__":
    main()
