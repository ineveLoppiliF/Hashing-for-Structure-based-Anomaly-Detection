from abc import ABC, abstractmethod
from numpy import exp, ndarray


class Preference(ABC):
    @staticmethod
    def invoke(preference_type: str) -> 'Preference':
        # Map preference type to a preference class
        preference_type_to_preference_map: dict = {'binary': Binary,
                                                   'gaussian': Gaussian}
        if preference_type not in preference_type_to_preference_map:
            raise ValueError('Bad preference type {}'.format(preference_type))
        return preference_type_to_preference_map[preference_type]

    @staticmethod
    @abstractmethod
    def compute(residuals: ndarray[float], sigma: float) -> ndarray:
        pass


class Binary(Preference):
    @staticmethod
    def compute(residuals: ndarray[float], sigma: float) -> ndarray[bool]:
        # Compute binary preferences
        preferences: ndarray[bool] = residuals <= sigma
        return preferences


class Gaussian(Preference):
    @staticmethod
    def compute(residuals: ndarray[float], sigma: float) -> ndarray[float]:
        # Compute gaussian preferences
        preferences: ndarray[float] = exp(-0.5 * (residuals / sigma)**2)
        return preferences
