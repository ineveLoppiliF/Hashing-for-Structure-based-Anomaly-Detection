from abc import ABC, abstractmethod
from numpy import ndarray
from numpy.random import choice


class Sampler(ABC):
    @staticmethod
    def create(sampling_type: str, **kwargs) -> 'Sampler':
        # Map sampling type to a sampler class
        sampling_type_to_class_map: dict = {'uniform': Uniform}
        if sampling_type not in sampling_type_to_class_map:
            raise ValueError('Bad sampling type {}'.format(sampling_type))
        return sampling_type_to_class_map[sampling_type](**kwargs)

    def __init__(self, n_samples: int, data: ndarray):
        self.n_samples: int = n_samples
        self.data: ndarray = data

    @abstractmethod
    def sample(self) -> ndarray:
        pass


class Uniform(Sampler):
    def sample(self) -> ndarray:
        return self.data[choice(self.data.shape[0], size=self.n_samples, replace=False)]
