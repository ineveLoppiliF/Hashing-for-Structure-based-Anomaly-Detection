from ..inode import INode
from numpy import ndarray


class VoronoiINode(INode):
	def __init__(self, data_size: int, children: ndarray[INode], depth: int, node_index: int,
				 split_points: ndarray):
		super().__init__(data_size, children, depth, node_index)
		self.split_points: ndarray = split_points
