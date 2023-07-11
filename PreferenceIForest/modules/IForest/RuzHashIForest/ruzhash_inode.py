from ..inode import INode
from numpy import ndarray


class RuzHashINode(INode):
    def __init__(self, data_size: int, children: ndarray[INode], depth: int, node_index: int,
                 thresholds: ndarray[float], components_permutation: ndarray[int], partition_labels: ndarray[int]):
        super().__init__(data_size, children, depth, node_index)
        self.thresholds: ndarray[float] = thresholds
        self.components_permutation: ndarray[int] = components_permutation
        self.partition_labels: ndarray[int] = partition_labels
