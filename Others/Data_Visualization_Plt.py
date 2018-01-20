import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR

names = ['X_acc', 'Y_acc', 'Z_acc', 'X_gyro', 'Y_gyro', 'Z_gyro']
dataset = pd.read_csv(
    'your_data.csv', header=None, names=names)
dataset['roll_mean_X_acc'] = dataset['X_acc'].rolling(window=50).mean()
dataset['roll_mean_Y_acc'] = dataset['Y_acc'].rolling(window=50).mean()
dataset['roll_mean_Z_acc'] = dataset['Z_acc'].rolling(window=50).mean()
dataset['roll_mean_X_gyro'] = dataset['X_gyro'].rolling(window=50).mean()
dataset['roll_mean_Y_gyro'] = dataset['Y_gyro'].rolling(window=50).mean()
dataset['roll_mean_Z_gyro'] = dataset['Z_gyro'].rolling(window=50).mean()
data = dataset.values
ar_coeffs = []
time_index = pd.to_datetime(range(len(data[49:, 0])),  unit='ms')
dataset
data[49:, 6]
for i in range(6, 12):
    series = pd.Series(data[49:, i], index=time_index)
    # Train AR model
    model = AR(series)
    model_fit = model.fit(maxlag=10)
    plt.plot(range(len(data[49:, i])), data[49:, i], 'b-', label='data')
    plt.plot(range(model_fit.k_ar, len(data[49:, i])), model_fit.fittedvalues, 'r-')
    plt.savefig("ar.png", dpi=300)
    plt.show()
    ar_coeffs = np.append(ar_coeffs, model_fit.params.values[1:])
ar_coeffs

dataset.index.shape
for axis in names:
    # Plot seperately
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot(range(len(dataset.index)), dataset[axis])
    axes[0].set_title(axis)
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('0.1mG / 0.01dps')
    axes[1].plot(range(len(dataset.index)), dataset['roll_mean_' + axis])
    axes[1].set_title(axis + ' SMA Plot')
    axes[1].set_xlabel('time')
    plt.savefig("sma.png", dpi=300)
    plt.show()

for axis in names:
    # Plot two on the same graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(dataset.index)), dataset['roll_mean_' + axis],
            color=(0, 0, 0), linewidth=4, alpha=.9, label='SMA')
    ax.plot(range(len(dataset.index)), dataset[axis], color=(1, 0, 0), label='Original')
    ax.set_title(axis)
    ax.set_xlabel('time')
    ax.set_ylabel('0.1mG / 0.01dps')
    ax.legend(loc='upper right')
    plt.show()


# Plot seperately
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].plot('time', 'X_acc', data=dataset)
axes[0].set_title('Original Plot')
axes[0].set_xlabel('time')
axes[0].set_ylabel('acc')
axes[1].plot('time', 'Rolling_Mean', data=dataset)
axes[1].set_title('SMA Plot')
axes[1].set_xlabel('time')
axes[1].set_ylabel('acc')
plt.show()
# Plot two on the same graph
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset['time'], dataset['Rolling_Mean'], color=(
    0, 0, 0), linewidth=4, alpha=.9, label='SMA')
ax.plot(dataset['time'], dataset['X_acc'], color=(1, 0, 0), label='Original')
ax.set_title('Original and SMA')
ax.set_xlabel('time')
ax.set_ylabel('acc')
ax.legend(loc='upper right')
plt.show()
