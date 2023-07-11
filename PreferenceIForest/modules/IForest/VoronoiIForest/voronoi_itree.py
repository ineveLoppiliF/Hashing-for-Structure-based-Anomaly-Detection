from ...utils import Distance
from ..itree import ITree
from ..inode import INode
from numpy import apply_along_axis, arange, asarray, empty, flatnonzero, full, int64, log, ndarray, split, where,\
                  zeros_like
from numpy.random import choice


class VoronoiITree(ITree):
	def __init__(self, branching_factor: int, metric: str = 'tanimoto'):
		super().__init__(branching_factor)
		self.metric: str = metric

	def build(self, data: ndarray) -> 'VoronoiITree':
		# Adjust depth limit according to data cardinality and branching factor
		self.depth_limit: float = log(data.shape[0]) / log(self.branching_factor)
		# Recursively build the tree
		self.nodes_count, self.root = self.recursive_build(data)
		return self

	def recursive_build(self, data: ndarray, depth: int = 0, node_index: int = 0) -> (int, INode):
		# If there aren't enough samples to be split according to the branching factor or the depth limit has been
		# reached, build a leaf node
		if data.shape[0] < self.branching_factor or depth >= self.depth_limit:
			return node_index + 1, INode.create('voronoiinode', data_size=data.shape[0], children=None, depth=depth,
												node_index=node_index, split_points=None)
		else:
			# Generate split points
			split_points: ndarray = data[choice(data.shape[0], size=self.branching_factor, replace=False)]
			# Partition data
			partition_indices: list[ndarray[int]] = self.split_data(data, split_points)
			# Generate recursively children nodes
			children: ndarray[INode] = empty(shape=(self.branching_factor,), dtype=INode)
			for i, indices in enumerate(partition_indices):
				node_index, children[i] = self.recursive_build(data[indices], depth + 1, node_index)
			return node_index + 1, INode.create('voronoiinode', data_size=data.shape[0], children=children, depth=depth,
												node_index=node_index, split_points=split_points)

	def predict(self, data: ndarray) -> ndarray[float]:
		# Compute depth of each sample
		return self.recursive_depth_search(self.root, data, empty(shape=(data.shape[0],), dtype=float))

	def recursive_depth_search(self, node: INode, data: ndarray, depths: ndarray[float]) -> ndarray[float]:
		# If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
		if node.children is None or data.shape[0] == 0:
			depths[:] = node.depth + ITree.get_random_path_length(self.branching_factor, node.data_size)
		else:
			# Partition data
			partition_indices: list[ndarray[int]] = self.split_data(data, node.split_points)
			# Fill the vector of depths
			for i, indices in enumerate(partition_indices):
				depths[indices]: ndarray[float] = self.recursive_depth_search(node.children[i], data[indices],
																			  depths[indices])
		return depths

	def split_data(self, data: ndarray, split_points: ndarray) -> list[ndarray[int]]:
		# Compute distances of data from split points
		distances: ndarray[float] = Distance.invoke(self.metric).compute(data, split_points)
		# Build full membership mask
		full_membership: ndarray[bool] = distances == distances.min(axis=1, keepdims=True)
		# Keep randomly one of the memberships for each sample
		membership: ndarray[bool] = zeros_like(full_membership)
		membership[arange(full_membership.shape[0]),
				   apply_along_axis(lambda x: choice(flatnonzero(x)), 1, full_membership)]: ndarray[bool] = True
		# Split data according to their membership
		row, col = where(membership.T)
		partition: list[ndarray[int]] = split(col, flatnonzero(row[1:] != row[:-1]) + 1)
		# TODO: Search for an approach that maintains the correspondence between seeds and generated nodes
		if len(partition) < split_points.shape[0]:
			partition.extend((split_points.shape[0] - len(partition)) * [asarray([], dtype=int64)])
		return partition

	def apply(self, data: ndarray) -> ndarray[int]:
		# Compute leaf index of each sample
		return self.recursive_index_search(self.root, data, empty(shape=(data.shape[0],), dtype=int))

	def recursive_index_search(self, node: INode, data: ndarray, leaves_index: ndarray[int]) -> ndarray[int]:
		# If the current node is a leaf, fill the leaves index vector with the current node index
		if node.children is None or data.shape[0] == 0:
			leaves_index[:] = node.node_index
		else:
			# Partition data
			partition_indices: list[ndarray[int]] = self.split_data(data, node.split_points)
			# Fill the vector of leaves index
			for i, indices in enumerate(partition_indices):
				leaves_index[indices]: ndarray[int] = self.recursive_index_search(node.children[i], data[indices],
																				  leaves_index[indices])
		return leaves_index

	def weight(self, data: ndarray) -> ndarray[int]:
		# Compute leaf mass of each sample
		return self.recursive_mass_search(self.root, data, empty(shape=(data.shape[0],), dtype=int))

	def recursive_mass_search(self, node: INode, data: ndarray, masses: ndarray[int]) -> ndarray[int]:
		# If the current node is a leaf, fill the masses vector with the current node mass
		if node.children is None or data.shape[0] == 0:
			masses[:] = node.data_size
		else:
			# Partition data
			partition_indices: list[ndarray[int]] = self.split_data(data, node.split_points)
			# Fill the vector of leaves index
			for i, indices in enumerate(partition_indices):
				masses[indices]: ndarray[int] = self.recursive_mass_search(node.children[i], data[indices],
																		   masses[indices])
		return masses

	def decision_path(self, data: ndarray) -> ndarray[bool]:
		# Compute path of each sample
		return self.recursive_path_search(self.root, data, full((data.shape[0], self.nodes_count), False))

	def recursive_path_search(self, node: INode, data: ndarray, paths: ndarray[bool]) -> ndarray[bool]:
		# Fill the position of the actual node
		paths[:, node.node_index] = True
		# If the current node is not a leaf, recursively search for paths
		if node.children is not None and data.shape[0] != 0:
			# Partition data
			partition_indices: list[ndarray[int]] = self.split_data(data, node.split_points)
			# Fill the vector of leaves index
			for i, indices in enumerate(partition_indices):
				paths[indices]: ndarray[bool] = self.recursive_path_search(node.children[i], data[indices],
																		   paths[indices])
		return paths
