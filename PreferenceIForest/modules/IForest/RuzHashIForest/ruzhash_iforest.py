from ...utils import Sampler
from ..iforest import IForest
from ..itree import ITree
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from numpy import asarray, ndarray
from sys import stdout
from tqdm import tqdm


class RuzHashIForest(IForest):
    def __init__(self, num_trees: int = 100, max_samples: int = 256, branching_factor: int = 2, n_jobs: int = 1):
        super().__init__(num_trees, max_samples, branching_factor, n_jobs)

    def fit(self, data: ndarray) -> 'RuzHashIForest':
        # Clean the tree list
        self.trees: list[ITree] = []
        # Adjust the number of samples to be picked according to data cardinality
        self.max_samples: int = min(self.max_samples, data.shape[0])
        # Adjust the branching factor according to max samples
        self.branching_factor: int = min(self.branching_factor, self.max_samples)
        # Compute the normalization factor
        self.normalization_factor: float = ITree.get_random_path_length(self.branching_factor, self.max_samples)
        # Instantiate a list of ITrees' build functions
        build_funcs: list['function'] = [ITree.create('ruzhashitree', branching_factor=self.branching_factor).build
                                         for _ in range(self.num_trees)]
        # Sample a list of max samples, one for each ITree
        sampler: Sampler = Sampler.create('uniform', n_samples=self.max_samples, data=data)
        sampled_data: list = [sampler.sample() for _ in range(self.num_trees)]
        # Build RuzHash ITrees
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.trees: list[ITree] = list(tqdm(executor.map(lambda f, x: f(x), build_funcs, sampled_data),
                                                total=self.num_trees,
                                                desc='     RuzHash Isolation Forest -> Fit',
                                                file=stdout))
        return self

    def score_samples(self, data: ndarray) -> ndarray[float]:
        # Collect ITrees' predict functions
        predict_funcs: list['function'] = [tree.predict for tree in self.trees]
        # Compute the depths of all samples in each tree
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            depths: ndarray[float] = asarray(list(tqdm(executor.map(lambda f, x: f(x), predict_funcs,
                                                                    repeat(data, self.num_trees)),
                                                       total=self.num_trees,
                                                       desc='     RuzHash Isolation Forest -> Score',
                                                       file=stdout))).T
        # Compute the mean depth of each sample along all trees
        mean_depths: ndarray[float] = depths.mean(axis=1)
        # Compute normalized mean depths
        normalized_mean_depths: ndarray[float] = 2 ** (-mean_depths / self.normalization_factor)
        return normalized_mean_depths

    # def weight_samples(self, data: ndarray) -> ndarray[float]:
    #     # Collect ITrees' weight functions
    #     weight_functions: list['function'] = [tree.weight for tree in self.trees]
    #     # Compute the weight of all samples in each tree
    #     with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
    #         weights: ndarray[int] = asarray(list(tqdm(executor.map(lambda f, x: f(x), weight_functions,
    #                                                                repeat(data, self.num_trees)),
    #                                                   total=self.num_trees,
    #                                                   desc='     RuzHash Isolation Forest -> Weight',
    #                                                   file=stdout))).T
    #     # Compute the mean weight of each sample along all trees
    #     mean_weights: ndarray[float] = weights.mean(axis=1)
    #     return mean_weights
    #
    # def apply(self, data: ndarray) -> ndarray[int]:
    #     # Collect ITrees' apply functions
    #     apply_functions: list['function'] = [tree.apply for tree in self.trees]
    #     # Compute the leaves index of all samples in each tree
    #     with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
    #         indices: ndarray[int] = asarray(list(tqdm(executor.map(lambda f, x: f(x), apply_functions,
    #                                                                repeat(data, self.num_trees)),
    #                                                   total=self.num_trees,
    #                                                   desc='     RuzHash Isolation Forest -> Apply',
    #                                                   file=stdout))).T
    #     return indices
    #
    # def get_decision_paths(self, data: ndarray) -> list[ndarray[bool]]:
    #     # Collect ITrees' decision path functions
    #     decision_path_functions: list['function'] = [tree.decision_path for tree in self.trees]
    #     # Compute the paths of all samples in each tree
    #     with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
    #         paths: list[ndarray[bool]] = list(tqdm(executor.map(lambda f, x: f(x), decision_path_functions,
    #                                                             repeat(data, self.num_trees)),
    #                                                total=self.num_trees,
    #                                                desc='     Ruzhash Isolation Forest -> Decision Path',
    #                                                file=stdout))
    #     return paths
