from ..utils import Sampler
from .models import Model
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
from numpy import asarray, ndarray
from sys import stdout
from tqdm import tqdm


class PreferenceEmbedding:
    def __init__(self, model_type: str = 'subsample', num_models: int = 1000, sampling_type: str = 'uniform',
                 mss: int = None, preference_type: str = 'gaussian', sigma: float = 1, n_jobs: int = 1):
        self.model_type: str = model_type
        self.num_models: int = num_models
        self.mss: int = mss
        self.sampling_type: str = sampling_type
        self.preference_type: str = preference_type
        self.sigma: float = sigma
        self.models: list[Model] = []
        self.n_jobs: int = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

    def fit(self, data: ndarray) -> 'PreferenceEmbedding':
        # Clean the model list
        self.models: list[Model] = []
        # Adjust the mss according to data cardinality
        self.mss: int = min(self.mss, data.shape[0])
        # Instantiate a list of models' fit functions
        fit_funcs: list['function'] = [Model.create(self.model_type, preference_type=self.preference_type,
                                                    sigma=self.sigma).fit
                                       for _ in range(self.num_models)]
        # Sample a list of mss, one for each model
        sampler: Sampler = Sampler.create(self.sampling_type, n_samples=self.mss, data=data)
        sampled_data: list[ndarray] = [sampler.sample() for _ in range(self.num_models)]
        # Fit models
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.models: list[Model] = list(tqdm(executor.map(lambda f, x: f(x), fit_funcs, sampled_data),
                                                 total=self.num_models, desc='     Preference Embedding -> Fit',
                                                 file=stdout))
        return self

    def transform(self, data: ndarray) -> ndarray:
        # Collect models' compute preferences functions
        preference_funcs: list['function'] = [model.compute_preferences for model in self.models]
        # Compute preference matrix
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            preference_matrix: ndarray = asarray(list(tqdm(executor.map(lambda f, x: f(x), preference_funcs,
                                                                        repeat(data, self.num_models)),
                                                           total=self.num_models,
                                                           desc='     Preference Embedding -> Transform',
                                                           file=stdout))).T
        return preference_matrix

    def get_residuals(self, data: ndarray) -> ndarray:
        # Collect models' compute residuals functions
        residual_funcs: list['function'] = [model.compute_residuals for model in self.models]
        # Compute preference matrix
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            residual_matrix: ndarray = asarray(list(tqdm(executor.map(lambda f, x: f(x), residual_funcs,
                                                                        repeat(data, self.num_models)),
                                                         total=self.num_models,
                                                         desc='     Preference Embedding -> Distances',
                                                         file=stdout))).T
        return residual_matrix
