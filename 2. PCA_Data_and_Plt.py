import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

posture_file_path_train = "your_train_data.csv"
posture_file_path_test = "your_test_data.csv"
train_features = 54
test_features = 54
posture_train_data = pd.read_csv(posture_file_path_train, header=None)[range(0, train_features)]
posture_test_data = pd.read_csv(posture_file_path_test, header=None)[range(0, test_features)]


# Plot pair plot
sns.set(style='whitegrid', context='notebook')

sns.pairplot(posture_train_data, size=2.5)
plt.tight_layout()
plt.savefig('your_pair_plt.png', dpi=72)
plt.show()

# Plot Correlation matrix
# Z-normalize data

sc = StandardScaler()
standard_posture_train_data = sc.fit_transform(posture_train_data)
standard_posture_test_data = sc.transform(posture_test_data)


# Estimate the correlation matrix
P = np.corrcoef(standard_posture_train_data.T)

sns.set(font_scale=1.5)
hm = sns.heatmap(P,
                 cbar=True,
                 square=True)

plt.tight_layout()
plt.savefig('your_correlation_matrix_heat_map.png', dpi=300)
plt.show()

sns.reset_orig()


# PCA on train and test data
pca = PCA(n_components=54)
standard_posture_train_data_pca = pca.fit_transform(standard_posture_train_data)
standard_posture_test_data_pca = pca.transform(standard_posture_test_data)
np.savetxt("your_train_pca_data.csv", standard_posture_train_data_pca, delimiter=",")
np.savetxt("your_test_pca_data.csv", standard_posture_test_data_pca, delimiter=",")
pca.explained_variance_ratio_

# plot train pca
plt.scatter(standard_posture_train_data_pca[:, 0], standard_posture_train_data_pca[:, 2])
plt.title('your_train_pca_data)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()
plt.savefig('your_train_pca_plt.png', dpi=300)
plt.show()

# plot test pca
plt.scatter(standard_posture_test_data_pca[:, 0], standard_posture_test_data_pca[:, 2])
plt.title('your_test_pca_data')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()
plt.savefig('your_test_pca_plt.png', dpi=300)
plt.show()
