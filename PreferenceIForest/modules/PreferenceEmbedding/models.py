from ..utils import Distance
from .preferences import Preference
from abc import ABC, abstractmethod
from numpy import asarray, diag, eye, full, hstack, median, ndarray, newaxis, ones, sqrt, std, zeros
from numpy.linalg import norm, inv, svd


class Model(ABC):
    @staticmethod
    def create(model_type: str, **kwargs) -> 'Model':
        # Map model type to a model class
        model_type_to_class_map: dict = {'subsample': Subsample,
                                         'line': Line,
                                         'plane': Plane,
                                         'circle': Circle,
                                         'fundamental': Fundamental,
                                         'homography': Homography}
        if model_type not in model_type_to_class_map:
            raise ValueError('Bad model type {}'.format(model_type))
        return model_type_to_class_map[model_type](**kwargs)

    @staticmethod
    def get_normalization_matrix(data: ndarray[float]) -> ndarray[float]:
        # Compute data centroid and scale
        centroid: ndarray[float] = data.sum(axis=0) / data.shape[0]  # compute data centroid
        scale: ndarray[float] = sqrt(2) / sqrt(((data - centroid) ** 2).sum(axis=1)).mean()  # compute data scale

        # Compute scaling matrix and translation vector
        S: ndarray[float] = diag(full(shape=(data.shape[1]), fill_value=scale))  # create scaling matrix
        t: ndarray[float] = -scale * centroid  # create translation vector

        # Compute transformation matrix
        T: ndarray[float] = eye(data.shape[1] + 1)  # initialize transformation matrix
        T[:S.shape[0], :S.shape[1]] = S  # add scaling
        T[:t.shape[0], -1] = t  # add translation
        return T
    
    def __init__(self, preference_type: str, sigma: float):
        self.preference_type: str = preference_type
        self.sigma: float = sigma

    @abstractmethod
    def fit(self, data: ndarray) -> 'Model':
        pass

    @abstractmethod
    def compute_residuals(self, data: ndarray, **kwargs) -> ndarray[float]:
        pass

    def compute_preferences(self, data: ndarray) -> ndarray:
        # Compute residuals
        residuals: ndarray[float] = self.compute_residuals(data)
        # Compute preferences
        preferences: ndarray = Preference.invoke(self.preference_type).compute(residuals, self.sigma)
        return preferences

    def compute_sigma(self, data: ndarray):
        # Estimate data noise standard deviation
        return std(self.compute_residuals(data), ddof=1)

    def compute_x84_sigma(self, data: ndarray, theta: float = 3.5):
        # Estimate data noise standard deviation using x84 approach
        residuals: ndarray[float] = self.compute_residuals(data)
        return theta/0.6745 * median(abs(residuals - median(residuals)))


class Subsample(Model):
    def __init__(self, preference_type: str, sigma: float):
        super().__init__(preference_type, sigma)
        self.samples: ndarray = None

    def fit(self, data: ndarray) -> Model:
        # For the subsample model the points themselves are the model
        self.samples: ndarray = data
        return self

    def compute_residuals(self, data: ndarray, metric: str = 'euclidean') -> ndarray[float]:
        # Compute distances between data and sampled points
        distances: ndarray[float] = Distance.invoke(metric).compute(data, self.samples)
        # For each datum keep the minimum distance wrt sampled points
        residuals: ndarray[float] = distances.min(axis=1)
        return residuals


class Line(Model):
    def __init__(self, preference_type: str, sigma: float):
        super().__init__(preference_type, sigma)
        self.C: ndarray[float] = None

    def fit(self, data: ndarray[float]) -> Model:
        # Fit line using SVD

        # Normalize data
        h_data: ndarray[float] = hstack([data, ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T: ndarray[float] = Model.get_normalization_matrix(data)  # compute transformation matrix
        n_data: ndarray[float] = (T @ h_data.T).T  # normalize data

        # Construct equations of the linear system
        A: ndarray[float] = n_data

        # Compute SVD
        u, d, v = svd(A)  # compute SVD
        C: ndarray[float] = v[-1]  # last row of V contains the solution

        # Construct conic matrix
        C: ndarray[float] = asarray([[0, 0, C[0]/2],
                                     [0, 0, C[1]/2],
                                     [C[0]/2, C[1]/2, C[2]]])

        # Extract line parameters
        self.C: ndarray[float] = T.T @ C @ T  # denormalize solution
        return self

    def compute_residuals(self, data: ndarray[float], **kwargs) -> ndarray[float]:
        # Compute point-line geometric distance for each datum
        h_data: ndarray[float] = hstack([data, ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        # Construct equations of the linear system
        A: ndarray[float] = (h_data[:, :, newaxis] @ h_data[:, newaxis, :]).reshape(data.shape[0], 9)
        return abs(A @ self.C.flatten()) / sqrt((self.C[0, 2] + self.C[2, 0])**2 + (self.C[1, 2] + self.C[2, 1])**2)


class Plane(Model):
    def __init__(self, preference_type: str, sigma: float):
        super().__init__(preference_type, sigma)
        self.Q: ndarray[float] = None

    def fit(self, data: ndarray[float]) -> Model:
        # Fit plane using SVD

        # Normalize data
        h_data: ndarray[float] = hstack([data, ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T: ndarray[float] = Model.get_normalization_matrix(data)  # compute transformation matrix
        n_data: ndarray[float] = (T @ h_data.T).T  # normalize data

        # Construct equations of the linear system
        A: ndarray[float] = n_data

        # Compute SVD
        u, d, v = svd(A)  # compute SVD
        C: ndarray[float] = v[-1]  # last row of V contains the solution

        # Construct quadric matrix
        Q: ndarray[float] = asarray([[0, 0, 0, C[0]/2],
                                     [0, 0, 0, C[1]/2],
                                     [0, 0, 0, C[2]/2],
                                     [C[0]/2, C[1]/2, C[2]/2, C[3]/2]])

        # Extract line parameters
        self.Q: ndarray[float] = T.T @ Q @ T  # denormalize solution
        return self

    def compute_residuals(self, data: ndarray[float], **kwargs) -> ndarray[float]:
        # Compute point-plane geometric distance for each datum
        h_data: ndarray[float] = hstack([data, ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        # Construct equations of the linear system
        A: ndarray[float] = (h_data[:, :, newaxis] @ h_data[:, newaxis, :]).reshape(data.shape[0], 16)
        return abs(A @ self.Q.flatten()) / sqrt((self.Q[0, 3] + self.Q[3, 0]) ** 2 + (self.Q[1, 3] + self.Q[3, 1]) ** 2 + (self.Q[2, 3] + self.Q[3, 2]) ** 2)


class Circle(Model):
    def __init__(self, preference_type: str, sigma: float):
        super().__init__(preference_type, sigma)
        self.C: ndarray[float] = None

    def fit(self, data: ndarray[float]) -> Model:
        # Fit circle using SVD

        # Normalize data
        h_data: ndarray[float] = hstack([data, ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T: ndarray[float] = Model.get_normalization_matrix(data)  # compute transformation matrix
        n_data: ndarray[float] = (T @ h_data.T).T  # normalize data

        # Construct equations of the linear system
        A: ndarray[float] = asarray([n_data[:, 0] ** 2 + n_data[:, 1] ** 2, n_data[:, 0], n_data[:, 1],
                                     ones(shape=data.shape[0])]).T

        # Compute SVD
        u, d, v = svd(A)  # compute SVD
        C: ndarray[float] = v[-1]  # last row of V contains the solution

        # Construct conic matrix
        C: ndarray[float] = asarray([[C[0], 0, C[1]/2],
                                     [0, C[0], C[2]/2],
                                     [C[1]/2, C[2]/2, C[3]]])

        # Extract circle parameters
        self.C: ndarray[float] = T.T @ C @ T  # denormalize solution
        return self

    def compute_residuals(self, data: ndarray[float], **kwargs) -> ndarray[float]:
        # Compute point-circle algebraic distance for each datum
        h_data: ndarray[float] = hstack([data, ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        # Construct equations of the linear system
        A: ndarray[float] = (h_data[:, :, newaxis] @ h_data[:, newaxis, :]).reshape(data.shape[0], 9)
        return abs(A @ self.C.flatten())

        ## Compute point-circle geometric distance for each datum
        #C: ndarray[float] = self.C/self.C[0, 0]
        #center: ndarray[float] = -C[2, :-1]  # compute circle center
        #radius: float = sqrt(center[0]**2 + center[1]**2 - C[2, 2])  # compute circle radius
        #return abs(norm(data - center, axis=1) - radius)


class Fundamental(Model):
    def __init__(self, preference_type: str, sigma: float):
        super().__init__(preference_type, sigma)
        self.F: ndarray[float] = None

    def fit(self, data: ndarray[float]) -> Model:
        # Fit fundamental matrix using SVD

        # Normalize data in the first view
        h_data_1: ndarray[float] = hstack([data[:, :2], ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T_1: ndarray[float] = Model.get_normalization_matrix(data[:, :2])  # compute transformation matrix
        n_data_1: ndarray[float] = (T_1 @ h_data_1.T).T  # normalize data

        # Normalize data in the second view
        h_data_2: ndarray[float] = hstack([data[:, 2:], ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T_2: ndarray[float] = Model.get_normalization_matrix(data[:, 2:])  # compute transformation matrix
        n_data_2: ndarray[float] = (T_2 @ h_data_2.T).T  # normalize data

        # Construct equations of the linear system
        A: ndarray[float] = (n_data_1[:, :, newaxis] @ n_data_2[:, newaxis, :]).reshape(data.shape[0], 9)

        # Compute SVD
        u, d, v = svd(A)  # compute SVD
        C: ndarray[float] = v[-1]  # last row of V contains the solution

        # Construct fundamental matrix
        F: ndarray[float] = C.reshape(3, 3)

        # Enforce rank(F)==2
        u, d, v = svd(F)
        d: ndarray[float] = diag(d)
        d[2, 2]: float = 0
        F: ndarray[float] = u @ d @ v.T

        # Extract fundamental matrix parameters
        self.F: ndarray[float] = T_2.T @ F @ T_1  # denormalize solution
        return self

    def compute_residuals(self, data: ndarray[float], **kwargs) -> ndarray[float]:
        # Compute matches-fundamental_matrix algebraic error for each datum
        h_data_1: ndarray[float] = hstack([data[:, :2], ones((data.shape[0], 1))])  # transform data in the first view to homogeneous coordinates
        h_data_2: ndarray[float] = hstack([data[:, 2:], ones((data.shape[0], 1))])  # transform data in the second view to homogeneous coordinates
        # Construct equations of the linear system
        A: ndarray[float] = (h_data_1[:, :, newaxis] @ h_data_2[:, newaxis, :]).reshape(data.shape[0], 9)
        return abs(A @ self.F.flatten())


class Homography(Model):
    @staticmethod
    def skew(u: ndarray[float]) -> ndarray[float]:
        # Compute the skew-symmetric cross-product matrix S such that u x v = S(u) @ v
        S: ndarray[float] = asarray([[zeros(shape=(u.shape[0],)), -u[:, 2], u[:, 1]],
                                     [u[:, 2], zeros(shape=(u.shape[0],)), -u[:, 0]],
                                     [-u[:, 1], u[:, 0], zeros(shape=(u.shape[0],))]])
        return S

    def __init__(self, preference_type: str, sigma: float):
        super().__init__(preference_type, sigma)
        self.H: ndarray[float] = None

    def fit(self, data: ndarray[float]) -> Model:
        # Fit homography matrix using SVD

        # Normalize data in the first view
        h_data_1: ndarray[float] = hstack([data[:, :2], ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T_1: ndarray[float] = Model.get_normalization_matrix(data[:, :2])  # compute transformation matrix
        n_data_1: ndarray[float] = (T_1 @ h_data_1.T).T  # normalize data

        # Normalize data in the second view
        h_data_2: ndarray[float] = hstack([data[:, 2:], ones((data.shape[0], 1))])  # transform data to homogeneous coordinates
        T_2: ndarray[float] = Model.get_normalization_matrix(data[:, 2:])  # compute transformation matrix
        n_data_2: ndarray[float] = (T_2 @ h_data_2.T).T  # normalize data

        # Construct equations of the linear system
        A: ndarray[float] = hstack((Homography.skew(n_data_2)[:, :, newaxis, :] * n_data_1.T[newaxis, :, :]).reshape(3, 9, data.shape[0])).T

        # Compute SVD
        u, d, v = svd(A)  # compute SVD
        C: ndarray[float] = v[-1]  # last row of V contains the solution

        # Construct homography matrix
        H: ndarray[float] = C.reshape(3, 3)

        # Extract homography matrix parameters
        self.H: ndarray[float] = inv(T_2) @ H @ T_1  # denormalize solution
        return self

    def compute_residuals(self, data: ndarray[float], **kwargs) -> ndarray[float]:
        # Compute matches-homography_matrix algebraic error for each datum
        h_data_1: ndarray[float] = hstack([data[:, :2], ones((data.shape[0], 1))])  # transform data in the first view to homogeneous coordinates
        h_data_2: ndarray[float] = hstack([data[:, 2:], ones((data.shape[0], 1))])  # transform data in the second view to homogeneous coordinates
        # Construct equations of the linear system
        A: ndarray[float] = (Homography.skew(h_data_2)[:, :, newaxis, :] * h_data_1.T[newaxis, :, :]).reshape(3, 9, data.shape[0])
        h: ndarray[float] = self.H.flatten()
        return norm((A * h[:, newaxis]).sum(axis=1), axis=0)
