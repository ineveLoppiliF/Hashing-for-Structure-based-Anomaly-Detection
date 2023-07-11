from abc import ABC, abstractmethod
from numpy import divide, dot, logical_or, maximum, minimum, ndarray, newaxis, sum, zeros_like
from scipy.spatial.distance import cdist


class Distance(ABC):
    @staticmethod
    def invoke(metric: str) -> 'Distance':
        # Map metric to a metric class
        metric_type_to_class_map: dict = {'tanimoto': Tanimoto,
                                          'ruzicka': Ruzicka,
                                          'euclidean': Scipy('euclidean'),
                                          'cityblock': Scipy('cityblock'),
                                          'seuclidean': Scipy('seuclidean'),
                                          'sqeuclidean': Scipy('sqeuclidean'),
                                          'cosine': Scipy('cosine'),
                                          'correlation': Scipy('correlation'),
                                          'hamming': Scipy('hamming'),
                                          'jaccard': Scipy('jaccard'),
                                          'jensenshannon': Scipy('jensenshannon'),
                                          'chebyshev': Scipy('chebyshev'),
                                          'canberra': Scipy('canberra'),
                                          'braycurtis': Scipy('braycurtis'),
                                          'mahalanobis': Scipy('mahalanobis'),
                                          'yule': Scipy('yule'),
                                          'matching': Scipy('matching'),
                                          'dice': Scipy('dice'),
                                          'kulczynski1': Scipy('kulczynski1'),
                                          'rogerstanimoto': Scipy('rogerstanimoto'),
                                          'russellrao': Scipy('russellrao'),
                                          'sokalmichener': Scipy('sokalmichener'),
                                          'sokalsneath': Scipy('sokalsneath')}
        if metric not in metric_type_to_class_map:
            raise ValueError('Bad metric {}'.format(metric))
        return metric_type_to_class_map[metric]

    @staticmethod
    @abstractmethod
    def compute(XA: ndarray, XB: ndarray) -> ndarray[float]:
        pass


class Scipy(Distance):
    def __init__(self, metric: str):
        self.metric: str = metric

    def compute(self, XA: ndarray, XB: ndarray) -> ndarray[float]:
        distances: ndarray[float] = cdist(XA, XB, metric=self.metric)
        return distances


class Tanimoto(Distance):
    @staticmethod
    def compute(XA: ndarray[float], XB: ndarray[float]) -> ndarray[float]:
        # Compute Tanimoto similarity for each couple of samples
        AB: ndarray[float] = dot(XA, XB.T)
        A_squared: ndarray[float] = sum(XA * XA, axis=1)
        B_squared: ndarray[float] = sum(XB * XB, axis=1)
        num: ndarray[float] = AB
        denom: ndarray[float] = A_squared[:, newaxis] + B_squared[newaxis, :] - AB
        Y: ndarray[float] = divide(num, denom, out=zeros_like(num, dtype=float), where=logical_or(num != 0, denom != 0))
        # Transform similarities into distances
        Y: ndarray[float] = 1 - Y
        return Y


class Ruzicka(Distance):
    @staticmethod
    def compute(XA: ndarray[float], XB: ndarray[float]) -> ndarray[float]:
        # Compute Ruzicka similarity for each couple of samples
        num: ndarray[float] = minimum(XA[:, newaxis], XB[newaxis, :]).sum(axis=2)
        denom: ndarray[float] = maximum(XA[:, newaxis], XB[newaxis, :]).sum(axis=2)
        Y: ndarray[float] = divide(num, denom, out=zeros_like(num, dtype=float), where=logical_or(num != 0, denom != 0))
        # Transform similarities into distances
        Y: ndarray[float] = 1 - Y
        return Y
