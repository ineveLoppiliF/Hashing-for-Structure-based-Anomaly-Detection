from .inode import INode
from abc import ABC, abstractmethod
from numpy import euler_gamma, log, ndarray


class ITree(ABC):
    @staticmethod
    def create(itree_type: str, **kwargs) -> 'ITree':
        # TODO: Find an alternative solution to overcome circular imports
        from .RuzHashIForest import RuzHashITree
        from .VoronoiIForest import VoronoiITree
        # Map itree type to an itree class
        itree_type_to_itree_map: dict = {'ruzhashitree': RuzHashITree,
                                         'voronoiitree': VoronoiITree}
        if itree_type not in itree_type_to_itree_map:
            raise ValueError('Bad itree type {}'.format(itree_type))
        return itree_type_to_itree_map[itree_type](**kwargs)

    @staticmethod
    def get_random_path_length(branching_factor: int, num_samples: int) -> float:
        if branching_factor == 2:
            if num_samples <= 1:
                return 0
            elif num_samples == 2:
                return 1
            else:
                return 2.0 * (log(num_samples - 1.0) + euler_gamma) - 2.0 * (num_samples - 1.0) / num_samples
        else:
            if num_samples <= 1:  # num_samples < branching_factor:
                return 0
            elif 1 < num_samples <= branching_factor:  # num_samples == branching_factor:
                return 1
            else:
                return log(num_samples) / log(
                    branching_factor)  # return (log(num_samples) + log(branching_factor - 1) + euler_gamma) / log(branching_factor) - 0.5

    def __init__(self, branching_factor: int):
        self.branching_factor: int = branching_factor
        self.depth_limit: float = 0.0
        self.root: INode = None
        self.nodes_count: int = 0

    @abstractmethod
    def build(self, data: ndarray) -> 'ITree':
        pass

    @abstractmethod
    def recursive_build(self, data: ndarray, depth: int = 0, node_index: int = 0) -> (int, INode):
        pass

    @abstractmethod
    def predict(self, data: ndarray) -> ndarray[float]:
        pass

    @abstractmethod
    def recursive_depth_search(self, node: INode, data: ndarray, depths: ndarray[float]) -> ndarray[float]:
        pass

    @abstractmethod
    def split_data(self, data: ndarray, **kwargs) -> list[ndarray[int]]:
        pass
