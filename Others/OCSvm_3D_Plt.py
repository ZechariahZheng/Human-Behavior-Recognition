import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from sklearn import svm
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

num_train_data = 9999
num_test_data = 9999

SPACE_SAMPLING_POINTS = 100

# Define the size of the space which is interesting for the example
X_MIN = -6
X_MAX = 13
Y_MIN = -6
Y_MAX = 7
Z_MIN = -5
Z_MAX = 5

# Generate a regular grid to sample the 3D space for various operations later
xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                         np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                         np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))

# Data
file_path_train = "your_train_data.csv"
file_path_test = "your_test_data.csv"

X_train = pd.read_csv(file_path_train, header=None)[[0, 1, 2]]
X_train.shape
X_test = pd.read_csv(file_path_test, header=None)[[0, 1, 2]]
X_test.shape


# Create a OneClassSVM instance and fit it to the data
clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.18)
clf.fit(X_train)


# Predict the class of the various input created before
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

predict_train_rate = y_pred_train[y_pred_train == 1].size
predict_test_rate = y_pred_test[y_pred_test == -1].size

y_pred_train[y_pred_train == 1].size / num_train_data
y_pred_test[y_pred_test == -1].size / num_test_data


# Calculate the distance from the separating hyperplane of the SVM for the
# whole space using the grid defined in the beginning
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Create a figure with axes for 3D plotting
fig = plt.figure()
ax = fig.gca(projection='3d')
fig.suptitle("Non Target Detection")

# Plot the different input points using 3D scatter plotting
b = ax.scatter(X_train.values[:, 0], X_train.values[:, 1], X_train.values[:, 2], c='black')
c = ax.scatter(X_test.values[:, 0], X_test.values[:, 1], X_test.values[:, 2], c='gold',)

# Plot the separating hyperplane by recreating the isosurface for the distance
# == 0 level in the distance grid computed through the decision function of the
# SVM. This is done using the marching cubes algorithm implementation from
# scikit-image.
verts, faces = measure.marching_cubes_classic(Z, 0)
# Scale and transform to actual size of the interesting volume
verts = verts * \
    [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
verts = verts + [X_MIN, Y_MIN, Z_MIN]
# and create a mesh to display
mesh = Poly3DCollection(verts[faces],
                        facecolor='orange', edgecolor='grey', alpha=0.3)
ax.add_collection3d(mesh)

# Some presentation tweaks
ax.set_xlim((X_MIN, X_MAX))
ax.set_ylim((Y_MIN, Y_MAX))
ax.set_zlim((Z_MIN, Z_MAX))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend([mpatches.Patch(color='orange', alpha=0.3), b, c],
          ["learned frontier", "training observations", "new abnormal observations"],
          loc="lower left",
          prop=matplotlib.font_manager.FontProperties(size=11))
ax.set_title(
    "Correct train target prediction: %d/" + num_train_data "\n"
    "Correct test non target prediction: %d/" + num_test_data
    % (predict_train_rate, predict_test_rate))
plt.show()
