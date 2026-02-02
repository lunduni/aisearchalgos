from searchalgorithms.base import SearchAlgorithmBase 

class astar(SearchAlgorithmBase):
    def __init__(self) -> None:
        super().__init__()
                        
    def reset(self, grid, start, goal): 
        super().reset(grid, start, goal)
        # If you want to initialize other stuff, put it here