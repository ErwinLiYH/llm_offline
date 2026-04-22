"""Quick planning probe for PointMaze checkpoints or cloud chat models.

This script isolates map understanding and route planning from low-level control.
It supports two modes:
  - path: prompt once and ask for the full path in {U,D,L,R}
  - step: prompt at each state and ask for the next move only

Both modes validate the produced route and compare it against the BFS shortest
path.

Example:
    micromamba run -n llm_offline python plan_probe.py \
        --model-path checkpoints/pointmaze/Qwen3-0.6B/.../ep1 \
        --variant large \
        --samples 20

    OPENAI_API_KEY=... micromamba run -n llm_offline python plan_probe.py \
        --provider openai \
        --api-model gpt-5-mini \
        --variant large \
        --samples 20
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import deque

import torch

from data.pointmaze.variants import POINTMAZE_VARIANTS
from model.policy import load_from_checkpoint

MOVE_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

PLANNING_PROMPT = """You are solving a maze routing task.

Maze name: {env_name}
Maze layout (# = wall, . = free):
{maze_visual}

Rows and columns are counted from the top-left corner starting at 1.
Start cell: row {start_row}, column {start_col}
Goal cell: row {goal_row}, column {goal_col}

Task: output a shortest valid path from the start cell to the goal cell.
Output format: a comma-separated sequence using only U,D,L,R.
Meanings: U=up, D=down, L=left, R=right.
Do not output any explanation.
If start equals goal, output STAY.
Path:
"""

PLANNING_PROMPT_WITH_REASONING = """You are solving a maze routing task.

Maze name: {env_name}
Maze layout (# = wall, . = free):
{maze_visual}

Rows and columns are counted from the top-left corner starting at 1.
Start cell: row {start_row}, column {start_col}
Goal cell: row {goal_row}, column {goal_col}

Think through the routing problem before answering.
Then output the final answer on the last line only.
Final answer format: PATH: followed by a comma-separated sequence using only U,D,L,R.
Meanings: U=up, D=down, L=left, R=right.
If start equals goal, output PATH: STAY on the last line.

Reasoning and final answer:
"""

STEP_PROMPT = """You are controlling an agent in a maze.

Maze name: {env_name}
Maze layout (# = wall, . = free):
{maze_visual}

Rows and columns are counted from the top-left corner starting at 1.
Current step: {step_index}
Current cell: row {current_row}, column {current_col}
Goal cell: row {goal_row}, column {goal_col}
Neighboring cells: up={up}, down={down}, left={left}, right={right}

Task: output exactly one next move that should help the agent reach the goal.
Valid outputs: U, D, L, R.
Meanings: U=up, D=down, L=left, R=right.
Do not output any explanation or extra text.
Move:
"""

STEP_PROMPT_WITH_REASONING = """You are controlling an agent in a maze.

Maze name: {env_name}
Maze layout (# = wall, . = free):
{maze_visual}

Rows and columns are counted from the top-left corner starting at 1.
Current step: {step_index}
Current cell: row {current_row}, column {current_col}
Goal cell: row {goal_row}, column {goal_col}
Neighboring cells: up={up}, down={down}, left={left}, right={right}

Think about the best next move before answering.
Then output the final answer on the last line only.
Final answer format: MOVE: followed by exactly one of U, D, L, R.
Do not put anything except that final answer on the last line.

Reasoning and final answer:
"""

STEP_HISTORY_BLOCK = """
Recent step history (oldest to newest):
{step_history}
"""


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="local", choices=["local", "openai"])
    parser.add_argument("--mode", type=str, default="path", choices=["path", "step"])
    parser.add_argument(
        "--direct-output",
        type=str_to_bool,
        default=True,
        help="true: answer only; false: allow reasoning and require the final answer on the last line.",
    )
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--api-model", type=str)
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--variant", type=str, default="large", choices=sorted(POINTMAZE_VARIANTS))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--start-row", type=int)
    parser.add_argument("--start-col", type=int)
    parser.add_argument("--goal-row", type=int)
    parser.add_argument("--goal-col", type=int)
    parser.add_argument(
        "--step-history-num",
        type=int,
        default=0,
        help="In step mode, include the most recent n executed steps in the prompt. 0 disables history.",
    )
    parser.add_argument(
        "--step-limit-factor",
        type=float,
        default=2.0,
        help="Maximum rollout steps in step mode as ceil(shortest_path_len * factor), with a minimum of 1.",
    )
    args = parser.parse_args()
    if args.provider == "local" and not args.model_path:
        parser.error("--model-path is required when --provider local")
    if args.provider == "openai" and not args.api_model:
        parser.error("--api-model is required when --provider openai")
    return args


def _free_cells(maze_map: list[list[int]]) -> list[tuple[int, int]]:
    cells = []
    for row, row_vals in enumerate(maze_map):
        for col, cell in enumerate(row_vals):
            if cell == 0:
                cells.append((row, col))
    return cells


def _neighbors(maze_map: list[list[int]], row: int, col: int):
    for move, (d_row, d_col) in MOVE_DELTAS.items():
        n_row = row + d_row
        n_col = col + d_col
        if 0 <= n_row < len(maze_map) and 0 <= n_col < len(maze_map[0]) and maze_map[n_row][n_col] == 0:
            yield move, n_row, n_col


def bfs_shortest_path(
    maze_map: list[list[int]], start: tuple[int, int], goal: tuple[int, int]
) -> list[str] | None:
    if start == goal:
        return []

    queue = deque([(start[0], start[1])])
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}

    while queue:
        row, col = queue.popleft()
        for move, n_row, n_col in _neighbors(maze_map, row, col):
            nxt = (n_row, n_col)
            if nxt in parents:
                continue
            parents[nxt] = ((row, col), move)
            if nxt == goal:
                path = []
                cur = nxt
                while parents[cur] is not None:
                    prev, prev_move = parents[cur]
                    path.append(prev_move)
                    cur = prev
                path.reverse()
                return path
            queue.append(nxt)
    return None


def sample_case(
    maze_map: list[list[int]], rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int], list[str]]:
    free_cells = _free_cells(maze_map)
    while True:
        start = rng.choice(free_cells)
        goal = rng.choice(free_cells)
        shortest = bfs_shortest_path(maze_map, start, goal)
        if shortest is not None:
            return start, goal, shortest


def build_prompt(meta: dict, start: tuple[int, int], goal: tuple[int, int]) -> str:
    prompt_vars = meta["prompt_vars"]
    template = PLANNING_PROMPT if meta.get("direct_output", True) else PLANNING_PROMPT_WITH_REASONING
    return template.format(
        env_name=prompt_vars["env_name"],
        maze_visual=prompt_vars["maze_visual"],
        start_row=start[0] + 1,
        start_col=start[1] + 1,
        goal_row=goal[0] + 1,
        goal_col=goal[1] + 1,
    )


def neighbor_status(maze_map: list[list[int]], row: int, col: int) -> dict[str, str]:
    statuses = {}
    for move, (d_row, d_col) in MOVE_DELTAS.items():
        n_row = row + d_row
        n_col = col + d_col
        blocked = (
            n_row < 0
            or n_row >= len(maze_map)
            or n_col < 0
            or n_col >= len(maze_map[0])
            or maze_map[n_row][n_col] == 1
        )
        label = {"U": "up", "D": "down", "L": "left", "R": "right"}[move]
        statuses[label] = "wall" if blocked else "free"
    return statuses


def build_step_prompt(
    meta: dict,
    current: tuple[int, int],
    goal: tuple[int, int],
    step_index: int,
    history_entries: list[dict] | None = None,
) -> str:
    prompt_vars = meta["prompt_vars"]
    statuses = neighbor_status(prompt_vars["maze_map"], current[0], current[1])
    template = STEP_PROMPT if meta.get("direct_output", True) else STEP_PROMPT_WITH_REASONING
    prompt = template.format(
        env_name=prompt_vars["env_name"],
        maze_visual=prompt_vars["maze_visual"],
        step_index=step_index,
        current_row=current[0] + 1,
        current_col=current[1] + 1,
        goal_row=goal[0] + 1,
        goal_col=goal[1] + 1,
        up=statuses["up"],
        down=statuses["down"],
        left=statuses["left"],
        right=statuses["right"],
    )
    if history_entries:
        history_lines = []
        for idx, entry in enumerate(history_entries, start=1):
            history_lines.append(
                "  "
                f"{idx}. start=(row {entry['start'][0] + 1}, col {entry['start'][1] + 1}), "
                f"goal=(row {entry['goal'][0] + 1}, col {entry['goal'][1] + 1}), "
                f"action={entry['move']}"
            )
        prompt += STEP_HISTORY_BLOCK.format(step_history="\n".join(history_lines))
    return prompt


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    new_tokens = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_text_openai(
    *,
    api_model: str,
    api_base: str | None,
    api_key_env: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Cloud mode requires the 'openai' package. Install it with "
            "`micromamba run -n llm_offline pip install openai`."
        ) from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable: {api_key_env}")

    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    try:
        response = client.responses.create(
            model=api_model,
            input=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.output_text
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code != 404:
            raise

    response = client.chat.completions.create(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def request_text(
    *,
    args,
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
) -> str:
    if args.provider == "local":
        return generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    return generate_text_openai(
        api_model=args.api_model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


def parse_moves(text: str) -> list[str]:
    stripped = text.strip().upper()
    if "PATH:" in stripped:
        stripped = stripped.split("PATH:")[-1].strip()
    if stripped.startswith("STAY"):
        return []
    return re.findall(r"[UDLR]", stripped)


def parse_single_move(text: str) -> str | None:
    stripped = text.strip().upper()
    if "MOVE:" in stripped:
        stripped = stripped.split("MOVE:")[-1].strip()
    moves = re.findall(r"[UDLR]", stripped)
    return moves[0] if moves else None


def evaluate_path(
    maze_map: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    moves: list[str],
) -> dict:
    row, col = start
    hit_wall = False
    for move in moves:
        d_row, d_col = MOVE_DELTAS[move]
        n_row = row + d_row
        n_col = col + d_col
        if not (0 <= n_row < len(maze_map) and 0 <= n_col < len(maze_map[0])) or maze_map[n_row][n_col] == 1:
            hit_wall = True
            break
        row, col = n_row, n_col

    ended_at = (row, col)
    reached_goal = ended_at == goal and not hit_wall
    return {
        "hit_wall": hit_wall,
        "ended_at": ended_at,
        "reached_goal": reached_goal,
        "path_length": len(moves),
    }


def evaluate_step_mode(
    *,
    args,
    meta: dict,
    model,
    tokenizer,
    device: torch.device,
    start: tuple[int, int],
    goal: tuple[int, int],
    shortest_len: int,
) -> dict:
    maze_map = meta["prompt_vars"]["maze_map"]
    current = start
    max_steps = max(1, int((shortest_len if shortest_len > 0 else 1) * args.step_limit_factor + 0.999999))
    raw_outputs: list[str] = []
    moves: list[str] = []
    history_entries: list[dict] = []
    invalid_output = False
    hit_wall = False

    for step_idx in range(1, max_steps + 1):
        if current == goal:
            break
        prompt_history = history_entries[-args.step_history_num :] if args.step_history_num > 0 else None
        prompt = build_step_prompt(meta, current, goal, step_idx, history_entries=prompt_history)
        generated = request_text(
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
        )
        raw_outputs.append(generated.strip())
        move = parse_single_move(generated)
        if move is None:
            invalid_output = True
            break

        moves.append(move)
        d_row, d_col = MOVE_DELTAS[move]
        n_row = current[0] + d_row
        n_col = current[1] + d_col
        if not (0 <= n_row < len(maze_map) and 0 <= n_col < len(maze_map[0])) or maze_map[n_row][n_col] == 1:
            hit_wall = True
            break
        history_entries.append(
            {
                "start": current,
                "goal": goal,
                "move": move,
            }
        )
        current = (n_row, n_col)

    reached_goal = current == goal and not hit_wall and not invalid_output
    return {
        "raw_outputs": raw_outputs,
        "moves": moves,
        "hit_wall": hit_wall,
        "invalid_output": invalid_output,
        "ended_at": current,
        "reached_goal": reached_goal,
        "path_length": len(moves),
        "max_steps": max_steps,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    meta = POINTMAZE_VARIANTS[args.variant]
    meta = {
        **meta,
        "direct_output": args.direct_output,
    }
    maze_map = meta["prompt_vars"]["maze_map"]

    model = None
    tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.provider == "local":
        print(f"[plan_probe] Using provider: local")
        print(f"[plan_probe] Using device: {device}")
        print(f"[plan_probe] Loading model from: {args.model_path}")
        model, tokenizer = load_from_checkpoint(
            args.model_path,
            load_in_4bit=True if args.load_in_4bit else None,
        )
        model.to(device)
        model.eval()
    else:
        print(f"[plan_probe] Using provider: openai")
        print(f"[plan_probe] API model: {args.api_model}")
        if args.api_base:
            print(f"[plan_probe] API base: {args.api_base}")
    print(f"[plan_probe] Mode: {args.mode}")
    print(f"[plan_probe] Direct output: {args.direct_output}")
    if args.mode == "step":
        print(f"[plan_probe] Step history num: {args.step_history_num}")

    explicit_case = all(
        value is not None
        for value in (args.start_row, args.start_col, args.goal_row, args.goal_col)
    )
    if explicit_case:
        start = (args.start_row - 1, args.start_col - 1)
        goal = (args.goal_row - 1, args.goal_col - 1)
        oracle = bfs_shortest_path(maze_map, start, goal)
        if oracle is None:
            raise ValueError("The provided start/goal pair is not connected.")
        cases = [(start, goal, oracle)]
    else:
        cases = [sample_case(maze_map, rng) for _ in range(args.samples)]

    results = []
    for idx, (start, goal, oracle) in enumerate(cases, start=1):
        shortest_len = len(oracle)
        print(
            f"[plan_probe] Case {idx}/{len(cases)} requesting"
            f" | start=({start[0] + 1},{start[1] + 1})"
            f" | goal=({goal[0] + 1},{goal[1] + 1})"
            f" | oracle_len={shortest_len}"
        )

        if args.mode == "path":
            prompt = build_prompt(meta, start, goal)
            generated = request_text(
                args=args,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
            )
            moves = parse_moves(generated)
            judged = evaluate_path(maze_map, start, goal, moves)
            shortest_reached = judged["reached_goal"] and judged["path_length"] == shortest_len
            result = {
                "reached_goal": judged["reached_goal"],
                "shortest": shortest_reached,
                "hit_wall": judged["hit_wall"],
                "invalid_output": False,
            }

            print(
                f"\nCase {idx}"
                f" | start=({start[0] + 1},{start[1] + 1})"
                f" | goal=({goal[0] + 1},{goal[1] + 1})"
                f" | oracle_len={shortest_len}"
            )
            print(f"Model raw output: {generated.strip()!r}")
            print(f"Parsed path: {','.join(moves) if moves else 'STAY'}")
            print(f"Oracle path: {','.join(oracle) if oracle else 'STAY'}")
            print(
                "Result:"
                f" reached_goal={judged['reached_goal']}"
                f" shortest={shortest_reached}"
                f" hit_wall={judged['hit_wall']}"
                f" ended_at=({judged['ended_at'][0] + 1},{judged['ended_at'][1] + 1})"
            )
        else:
            judged = evaluate_step_mode(
                args=args,
                meta=meta,
                model=model,
                tokenizer=tokenizer,
                device=device,
                start=start,
                goal=goal,
                shortest_len=shortest_len,
            )
            shortest_reached = judged["reached_goal"] and judged["path_length"] == shortest_len
            result = {
                "reached_goal": judged["reached_goal"],
                "shortest": shortest_reached,
                "hit_wall": judged["hit_wall"],
                "invalid_output": judged["invalid_output"],
            }

            print(
                f"\nCase {idx}"
                f" | start=({start[0] + 1},{start[1] + 1})"
                f" | goal=({goal[0] + 1},{goal[1] + 1})"
                f" | oracle_len={shortest_len}"
                f" | max_steps={judged['max_steps']}"
            )
            for step_i, raw in enumerate(judged["raw_outputs"], start=1):
                parsed = parse_single_move(raw)
                print(f"Step {step_i}: raw={raw!r} parsed={parsed}")
            print(f"Executed path: {','.join(judged['moves']) if judged['moves'] else 'STAY'}")
            print(f"Oracle path: {','.join(oracle) if oracle else 'STAY'}")
            print(
                "Result:"
                f" reached_goal={judged['reached_goal']}"
                f" shortest={shortest_reached}"
                f" hit_wall={judged['hit_wall']}"
                f" invalid_output={judged['invalid_output']}"
                f" ended_at=({judged['ended_at'][0] + 1},{judged['ended_at'][1] + 1})"
            )

        results.append(result)

    reached = sum(item["reached_goal"] for item in results)
    shortest = sum(item["shortest"] for item in results)
    wall_hits = sum(item["hit_wall"] for item in results)
    invalid_outputs = sum(item["invalid_output"] for item in results)
    total = len(results)

    print("\nSummary")
    print(f"  total_cases={total}")
    print(f"  reached_goal={reached}/{total} ({reached / total:.1%})")
    print(f"  shortest_path={shortest}/{total} ({shortest / total:.1%})")
    print(f"  hit_wall={wall_hits}/{total} ({wall_hits / total:.1%})")
    print(f"  invalid_output={invalid_outputs}/{total} ({invalid_outputs / total:.1%})")


if __name__ == "__main__":
    main()
