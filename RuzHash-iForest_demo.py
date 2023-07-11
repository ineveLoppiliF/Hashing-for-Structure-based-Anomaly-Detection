from matplotlib import pyplot as plt
from numpy import eye, linspace, meshgrid, vstack
from numpy.random import normal, uniform
from PreferenceIForest import PreferenceIForest
from PreferenceIForest.modules.utils import Distance
from sklearn.manifold import MDS

#%% Define execution parameters
pif_params: dict = {'model_type': 'line',
                    'num_models': 1000,
                    'sampling_type': 'uniform',
                    'mss': 2,
                    'preference_type': 'gaussian',
                    'sigma': 0.025,
                    'iforest_type': 'ruzhashiforest',
                    'num_trees': 100,
                    'max_samples': 256,
                    'branching_factor': 2,
                    'n_jobs': 1}
cardA = 5
card = 50
gauss1 = [[0.5, 0.85], 0.015]
gauss2 = [[0.15, 0.6], 0.015]
gauss3 = [[0.65, 0.65], 0.015]
line1 = [0.7, 0]
line2 = [-1.2, 1]

#%% Generate and plot synthetic dataset
data1 = vstack([normal(loc=gauss1[0][0], scale=gauss1[1], size=(cardA,)), normal(loc=gauss1[0][1], scale=gauss1[1], size=(cardA,))]).T
data2 = vstack([normal(loc=gauss2[0][0], scale=gauss2[1], size=(cardA,)), normal(loc=gauss2[0][1], scale=gauss2[1], size=(cardA,))]).T
data3 = vstack([normal(loc=gauss3[0][0], scale=gauss3[1], size=(cardA,)), normal(loc=gauss3[0][1], scale=gauss3[1], size=(cardA,))]).T

data4_x = uniform(0, 1, size=card)
data4 = vstack([data4_x + normal(loc=0, scale=0.01, size=card), data4_x*line1[0]+line1[1]] + normal(loc=0, scale=0.01, size=card)).T

data5_x = uniform(0, 1, size=card)
data5 = vstack([data5_x + normal(loc=0, scale=0.01, size=card), data5_x*line2[0]+line2[1]] + normal(loc=0, scale=0.01, size=card)).T


plt.rcParams["figure.figsize"] = (6, 6)
plt.figure()
plt.scatter(data1[:, 0], data1[:, 1], c='#FF6969', marker='.', s=80)
plt.scatter(data2[:, 0], data2[:, 1], c='#FF6969', marker='.', s=80)
plt.scatter(data3[:, 0], data3[:, 1], c='#FF6969', marker='.', s=80)
plt.scatter(data4[:, 0], data4[:, 1], c='#78D695', marker='.', s=80)
plt.scatter(data5[:, 0], data5[:, 1], c='#78D695', marker='.', s=80)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.show()

#%% Execute RuzHash-iForest and plot heatmap of Anomaly Scores
pif = PreferenceIForest(**pif_params)
pif.fit(vstack([data1, data2, data3, data4, data5]))

x = linspace(0, 1, 100)
y = linspace(0, 1, 100)
X, Y = meshgrid(x, y)
coords = vstack([X.ravel(), Y.ravel()]).T
scores = pif.score_samples(coords)

plt.rcParams["figure.figsize"] = (8, 6)
plt.figure()
plt.scatter(coords[:, 0], coords[:, 1], c=scores, cmap='rainbow', alpha=0.8, marker='.', s=120,
            edgecolors='none')
cbar = plt.colorbar(pad=0.025)
plt.scatter(data1[:, 0], data1[:, 1], c='#FF6969', marker='.', s=80)
plt.scatter(data2[:, 0], data2[:, 1], c='#FF6969', marker='.', s=80)
plt.scatter(data3[:, 0], data3[:, 1], c='#FF6969', marker='.', s=80)
plt.scatter(data4[:, 0], data4[:, 1], c='#78D695', marker='.', s=80)
plt.scatter(data5[:, 0], data5[:, 1], c='#78D695', marker='.', s=80)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks([], [])
plt.yticks([], [])
cbar.set_ticks([])
plt.tight_layout()
plt.show()

#%% Compute Ruzicka distances in the Preference Space and visualize them via Multi-Dimensional Scaling
all_data = pif.preference_embedding.transform(vstack([data1, data2, data3, data4, data5]))
all_distances = Distance.invoke('ruzicka').compute(all_data, all_data)
all_distances[eye(all_distances.shape[0], dtype=bool)] = 0
mds = MDS(n_components=2, metric=True, n_init=100, max_iter=1000, eps=1, dissimilarity='precomputed')
all_data_mds = mds.fit_transform(all_distances)

plt.rcParams["figure.figsize"] = (6, 6)
plt.figure()
plt.scatter(all_data_mds[:5, 0], all_data_mds[:5, 1], c='#FF6969', marker='.', s=80)
plt.scatter(all_data_mds[5:10, 0], all_data_mds[5:10, 1], c='#FF6969', marker='.', s=80)
plt.scatter(all_data_mds[10:15, 0], all_data_mds[10:15, 1], c='#FF6969', marker='.', s=80)
plt.scatter(all_data_mds[15:65, 0], all_data_mds[15:65, 1], c='#78D695', marker='.', s=80)
plt.scatter(all_data_mds[65:115, 0], all_data_mds[65:115, 1], c='#78D695', marker='.', s=80)
plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.show()
