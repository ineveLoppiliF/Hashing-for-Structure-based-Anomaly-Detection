from ..itree import ITree
from ..inode import INode
from numpy import apply_along_axis, arange, empty, hstack, log, ndarray, newaxis, unique, where
from numpy.random import choice, permutation, random


class RuzHashITree(ITree):
    def __init__(self, branching_factor: int):
        super().__init__(branching_factor)

    def build(self, data: ndarray) -> 'RuzHashITree':
        # Adjust depth limit according to data cardinality and branching factor
        self.depth_limit: float = log(data.shape[0]) / log(self.branching_factor)
        # Recursively build the tree
        self.nodes_count, self.root = self.recursive_build(data)
        return self

    def recursive_build(self, data: ndarray, depth: int = 0, node_index: int = 0) -> (int, INode):
        # If there aren't enough samples to be split according to the branching factor or the depth limit has been
        # reached, build a leaf node
        if data.shape[0] < self.branching_factor or depth >= self.depth_limit:
            return node_index + 1, INode.create('ruzhashinode', data_size=data.shape[0], children=None, depth=depth,
                                                node_index=node_index, thresholds=None, components_permutation=None,
                                                partition_labels=None)
        else:
            # Generate thresholds vector
            thresholds: ndarray[float] = random(data.shape[1])
            # Generate components permutation
            components_permutation: ndarray[int] = permutation(data.shape[1])
            # Generate partition labels
            partition_labels: ndarray[int] = permutation(hstack([arange(self.branching_factor),  # ensure to have non-empty subsets
                                                                 choice(self.branching_factor,
                                                                        size=data.shape[1] - self.branching_factor)]))
            # Partition data
            partition_indices: list[ndarray[int]] = self.split_data(data, thresholds, components_permutation,
                                                                    partition_labels)
            # Generate recursively children nodes
            children: ndarray[INode] = empty(shape=(self.branching_factor + 1,), dtype=INode)  # additional sink node
            for i, indices in enumerate(partition_indices[:-1]):
                node_index, children[i] = self.recursive_build(data[indices], depth + 1, node_index)
            # Generate sink node
            node_index, children[-1] = node_index + 1, INode.create('ruzhashinode', data_size=data[partition_indices[-1]].shape[0],
                                                                    children=None, depth=depth, node_index=node_index, thresholds=None,
                                                                    components_permutation=None, partition_labels=None)
            return node_index + 1, INode.create('ruzhashinode', data_size=data.shape[0], children=children, depth=depth,
                                                node_index=node_index, thresholds=thresholds,
                                                components_permutation=components_permutation,
                                                partition_labels=partition_labels)

    def predict(self, data: ndarray) -> ndarray[float]:
        # Compute depth of each sample
        return self.recursive_depth_search(self.root, data, empty(shape=(data.shape[0],), dtype=float))

    def recursive_depth_search(self, node: INode, data: ndarray, depths: ndarray[float]) -> ndarray[float]:
        # If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
        if node.children is None or data.shape[0] == 0:
            depths[:] = node.depth + ITree.get_random_path_length(self.branching_factor, node.data_size)
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self.split_data(data, node.thresholds, node.components_permutation,
                                                                    node.partition_labels)
            # Fill the vector of depths
            for i, indices in enumerate(partition_indices):
                depths[indices]: ndarray[float] = self.recursive_depth_search(node.children[i], data[indices],
                                                                              depths[indices])
        return depths

    def split_data(self, data: ndarray, thresholds: ndarray[float], components_permutation: ndarray[int],
                   partition_labels: ndarray[int]) -> list[ndarray[int]]:
        # Binarize data via thresholds vector
        data_binarized: ndarray[bool] = thresholds < data
        # Create mask of all-zeros binarized data
        full_zeros_mask: ndarray[bool] = ~data_binarized.any(axis=1)
        # Partition non all-zeros binarized data, if they exist
        if ~full_zeros_mask.all():
            # For each non all-zeros data, compute its minimum in the permutation
            data_minimum: ndarray[int] = apply_along_axis(lambda x, y: y[x].min(), 1, data_binarized[~full_zeros_mask],
                                                          components_permutation)
            # Assign non all-zeros data to their element of the partition
            data_partition: ndarray[int] = partition_labels[where(data_minimum[:, newaxis] == components_permutation)[1]]
        # Create an empty partition, if non all-zeros binarized data don't exist
        else:
            # Create an empty partition
            data_partition: ndarray[int] = empty(shape=(0,), dtype=int)
        # Split data according to their membership
        partition: list[ndarray[int]] = []
        for i in unique(partition_labels):
            partition.append(where(data_partition == i)[0])
        # Add all-zeros data partition
        partition.append(where(full_zeros_mask)[0])
        return partition
