from searchalgorithms.base import SearchAlgorithmBase


def _manhattan(a, b):
    # Calculate the Manhattan distance between two points a and b, 
    # where a and b are (row, col) tuples
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class gbfs(SearchAlgorithmBase):
    # Greedy Best-First Search algorithm implementation
    def __init__(self) -> None:
        super().__init__()
           
    def reset(self, grid, start, goal): 
        # Initialize the GBFS algorithm with the given grid, start, and goal positions
        super().reset(grid, start, goal)
        self._came_from = {start: None} if start else {}
        self._explored_set = set()
        self._frontier_set = {start} if start else set()
        if start and goal:
            self._frontier = [(start, _manhattan(start, goal), 0, start)]

    def getFrontier(self) -> list:
        # Return a list of nodes currently in the frontier (for visualization purposes)
        return [item[0] for item in self._frontier]

    def _neighbors(self, node):
        # Generate valid neighboring nodes (up, down, left, right) that are not walls (value 1 in the grid)
        r, c = node
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self._grid.shape[0] and 0 <= nc < self._grid.shape[1]:
                if self._grid[nr][nc] != 1:
                    yield (nr, nc)

    def _reconstruct_path(self, end_node):
        # Reconstruct the path from the start node to the given end node
        path = []
        cur = end_node
        while cur is not None:
            path.append(cur)
            cur = self._came_from.get(cur)
        path.reverse()
        return path

    def _update_metrics(self):
        # Update the metrics for expanded nodes, max depth, max frontier size, and max memory usage
        self._max_frontier_size = max(self._max_frontier_size, len(self._frontier))
        self._max_nodes_in_memory = max(self._max_nodes_in_memory, len(self._frontier) + len(self._explored))

    def _pop_lowest_h(self):
        # Pop the node from the frontier with the lowest h(n) value
        best_idx = 0
        best_h = self._frontier[0][1]
        for i in range(1, len(self._frontier)):
            h = self._frontier[i][1]
            if h < best_h:
                best_h = h
                best_idx = i
        return self._frontier.pop(best_idx)

    def step(self):
        # Perform one step of the GBFS algorithm
        if self._done:
            return

        if not self._frontier:
            self._done = True
            self._path = []
            self._cost = 0
            return

        node, h, g, parent = self._pop_lowest_h()
        self._frontier_set.discard(node)

        if node in self._explored_set:
            self._update_metrics()
            return

        self._explored.append(node)
        self._explored_set.add(node)

        if node == self._goal:
            self._path = self._reconstruct_path(node)
            self._cost = g
            self._done = True
            self._update_metrics()
            return

        for nbr in self._neighbors(node):
            if nbr in self._explored_set or nbr in self._frontier_set:
                continue
            self._came_from[nbr] = node
            depth = self._depth_map.get(node, 0) + 1
            self._depth_map[nbr] = depth
            self._max_depth = max(self._max_depth, depth)
            hn = _manhattan(nbr, self._goal)
            self._frontier.append((nbr, hn, g + 1, node))
            self._frontier_set.add(nbr)

        self._update_metrics()
         