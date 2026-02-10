# simple sanity check runner that runs the algorithm without a pygame window and prints out the results

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    # models the Results of running the algorithm
    cost: int
    expanded: int
    max_depth: int
    max_memory: int
    max_frontier: int
    path_len: int


def _find_unique_cell(grid: np.ndarray, value: float) -> tuple[int, int]:
    # find the unique cell in the grid with the given value, and return its (row, col) coordinates
    locs = np.argwhere(grid == value)
    if len(locs) != 1:
        raise ValueError(f"Expected exactly 1 cell with value={value}, found {len(locs)}")
    r, c = locs[0]
    return int(r), int(c)


def run_algorithm(algorithm_name: str, maze_id: int) -> Result:
    #  dynamically import the algorithm module and class based on the algorithm name
    module = __import__(f"searchalgorithms.{algorithm_name}")
    algo_mod = getattr(module, algorithm_name)
    algo_cls = getattr(algo_mod, algorithm_name)

    grid = np.loadtxt(f"mazes/maze{maze_id}.txt")
    start = _find_unique_cell(grid, 2)
    goal = _find_unique_cell(grid, 3)

    algo = algo_cls()
    algo.reset(grid, start, goal)

    for _ in range(1_000_000):
        if algo.isDone():
            break
        algo.step()
    else:
        raise TimeoutError("Algorithm did not terminate within 1,000,000 steps")

    return Result(
        cost=algo.getCost(),
        expanded=algo.getNumberOfExpanded(),
        max_depth=algo.getMaxDepth(),
        max_memory=algo.getMaxMemoryUsage(),
        max_frontier=algo.getMaxFrontierSize(),
        path_len=len(algo.getPath()),
    )


def main() -> None:
    # parse command line arguments for algorithm and maze id
    parser = argparse.ArgumentParser(description="Headless sanity runner (no pygame window)")
    parser.add_argument("--algo", required=True, choices=["bfs", "dfs", "ucs", "gbfs", "astar"])
    parser.add_argument("--maze_id", type=int, required=True)
    args = parser.parse_args()

    res = run_algorithm(args.algo, args.maze_id)
    print(f"algo={args.algo} maze={args.maze_id}")
    print(f"cost={res.cost}")
    print(f"expanded={res.expanded}")
    print(f"max_depth={res.max_depth}")
    print(f"max_frontier={res.max_frontier}")
    print(f"max_memory={res.max_memory}")
    print(f"path_len={res.path_len}")


if __name__ == "__main__":
    main()
