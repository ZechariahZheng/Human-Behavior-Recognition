import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt


df = pd.read_csv(
    'your_data.csv', header=None)
data = df.values
time_axis = data[:, 0]
data_y = data[:, 1]

params = stats.linregress(time_axis, data[:, 1])

plt.scatter(time_axis, data_y, c='#ffae3e')
plt.plot(time_axis, time_axis * params[0] + params[1], c='#ff4a30')
plt.savefig("your_beh.png", dpi=300)
plt.title("your_beh")
plt.show()
