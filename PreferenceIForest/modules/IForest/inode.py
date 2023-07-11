from abc import ABC
from numpy import ndarray


class INode(ABC):
    @staticmethod
    def create(inode_type: str, **kwargs) -> 'INode':
        # TODO: Find an alternative solution to overcome circular imports
        from .RuzHashIForest import RuzHashINode
        from .VoronoiIForest import VoronoiINode
        # Map inode type to an inode class
        inode_type_to_inode_map: dict = {'ruzhashinode': RuzHashINode,
                                         'voronoiinode': VoronoiINode}
        if inode_type not in inode_type_to_inode_map:
            raise ValueError('Bad inode type {}'.format(inode_type))
        return inode_type_to_inode_map[inode_type](**kwargs)

    def __init__(self, data_size: int, children: ndarray['INode'], depth: int, node_index: int):
        self.data_size: int = data_size
        self.children: ndarray['INode'] = children
        self.depth: int = depth
        self.node_index: int = node_index
