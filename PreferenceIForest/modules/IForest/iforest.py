from .itree import ITree
from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from numpy import ndarray


class IForest(ABC):
    @staticmethod
    def create(iforest_type: str, **kwargs) -> 'IForest':
        #TODO: Find an alternative solution to overcome circular imports
        from .RuzHashIForest import RuzHashIForest
        from .VoronoiIForest import VoronoiIForest
        # Map iforest type to an iforest class
        iforest_type_to_iforest_map: dict = {'ruzhashiforest': RuzHashIForest,
                                             'voronoiiforest': VoronoiIForest}
        if iforest_type not in iforest_type_to_iforest_map:
            raise ValueError('Bad iforest type {}'.format(iforest_type))
        return iforest_type_to_iforest_map[iforest_type](**kwargs)

    def __init__(self, num_trees: int = 100, max_samples: int = 256, branching_factor: int = 2, n_jobs: int = 1):
        self.num_trees: int = num_trees
        self.max_samples: int = max_samples
        self.branching_factor: int = branching_factor
        self.trees: list[ITree] = []
        self.normalization_factor: float = None
        self.n_jobs: int = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

    @abstractmethod
    def fit(self, data: ndarray) -> 'IForest':
        pass

    @abstractmethod
    def score_samples(self, data: ndarray) -> ndarray[float]:
        pass
