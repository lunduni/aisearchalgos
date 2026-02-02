class SearchAlgorithmBase:

    def __init__(self) -> None:
        # Initialize with empty/None values
        self.reset(None, None, None)
        
    def reset(self, grid, start, goal):
        """Set all internal variables to their initial values. """
        self._grid = grid  # The grid environment for the search algorithm has a size of (33, 33), which corresponds to (row, column) order.
        self._start = start # Starting Cell (row, column)
        self._goal = goal # Goal Cell (row, column)
        
        # Core Search Structures
        self._frontier = [(start, 0, 0, start)] # Frontier set for search algorithms. Frontier tuple: (node, heuristic_h, cost_g, parent)
        self._explored = []  # Using a set for faster lookups, keeping the explored set
        self._path = []  # To store the path
        self._depth_map = {start: 0} if start else {} # To store the depth  
        
        # Performance and state metrics
        self._done = False # The state of the search, `True` if the search terminated
        self._cost = 0  # Calculated cost of the solution path
        self._max_depth = 0 # Track the maximum depth reached 
        self._max_nodes_in_memory = 0  # Tracker for total memory footprint
        self._max_frontier_size = 0 # Max size of the frontier
         
    # --- Search Control ---
    def step(self):
        """Runs the algorithm for one step. 
        The loop for searching the solution is defined outside of the algorithm for visualization purposes.
        Step method is called until isDone method return True. 
        Must be implemented by child classes (e.g., BFS, DFS, A*)."""
        
        raise NotImplementedError("Subclasses must implement the step() method.")
                           
    def isDone(self) -> bool:
        """Returns True if the search has terminated (success or failure)."""
        return self._done          
    
    # --- Getters for Visualization ---    
    def getFrontier(self) -> list:
        """Returns the frontier set of your search algorithm for visualization purposes. """
        return self._frontier
    
    def getExplored(self) -> list:
        """Returns the explored set of your search algorithm for visualization purposes. """
        return self._explored
    
    def getPath(self) -> list:  
        """Returns the found path when the search algorithm terminates for visualization purposes. """
        return self._path
    
    def getCost(self) -> int:
        """Returns the cost of the path found by the search algorithm. 
        Note that the cost would be the lenght of the path but in some applications the cost may not be the case with constraints other than move cost. """
        return self._cost
    
    def getMaxFrontierSize(self) -> int:
        """Returns the max frontier size. """
        return self._max_frontier_size
    
    def getMaxMemoryUsage(self) -> int:
        """Returns the used node memory size. """
        return self._max_nodes_in_memory
    
    def getNumberOfExpanded(self) -> int:
        """ Returns the number of expanded nodes which is the lenght of the explored set. """
        return len(self._explored)
    
    def getMaxDepth(self) -> int:
        """Returns the maximum depth reached during the search."""
        return self._max_depth
    
    