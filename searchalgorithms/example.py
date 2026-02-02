from searchalgorithms.base import SearchAlgorithmBase
import numpy

class example(SearchAlgorithmBase):
    def __init__(self) -> None:
        super().__init__()

    def step(self):
        self._done = True