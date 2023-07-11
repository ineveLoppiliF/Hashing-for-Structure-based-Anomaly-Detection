from .modules import IForest, PreferenceEmbedding
from numpy import ndarray


class PreferenceIForest:
    def __init__(self, model_type: str = 'subsample', num_models: int = 1000, sampling_type: str = 'uniform',
                 mss: int = None, preference_type: str = 'gaussian', sigma: float = 1,
                 iforest_type: str = 'voronoiiforest', num_trees: int = 100, max_samples: int = 256,
                 branching_factor: int = 2, n_jobs: int = 1, **kwargs):
        self.preference_embedding: PreferenceEmbedding = PreferenceEmbedding(model_type, num_models, sampling_type, mss,
                                                                             preference_type, sigma, n_jobs)
        self.iforest: IForest = IForest.create(iforest_type, num_trees=num_trees, max_samples=max_samples,
                                               branching_factor=branching_factor, n_jobs=n_jobs, **kwargs)

    def fit(self, data: ndarray) -> 'PreferenceIForest':
        print('Preference Isolation Forest -> Fit')
        # Fit models on data for Preference Embedding
        self.preference_embedding.fit(data)
        # Transform data through Preference Embedding
        preference_matrix: ndarray = self.preference_embedding.transform(data)
        # Build IForest from preference matrix
        self.iforest.fit(preference_matrix)
        return self

    def score_samples(self, data: ndarray) -> ndarray[float]:
        print('Preference Isolation Forest -> Score')
        # Transform data through Preference Embedding
        preference_matrix: ndarray = self.preference_embedding.transform(data)
        # Score samples through IForest
        scores: ndarray[float] = self.iforest.score_samples(preference_matrix)
        return scores
