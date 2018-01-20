import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
import numpy as np

df = pd.read_csv(
    'your_data.csv', header=None)
data = df.values
data
time_axis = range(len(data[:, 0]))
ar_coeffs = []
for i in range(1, 4):
    series = pd.Series(data[:, i], index=time_axis)
    series.index = pd.to_datetime(series.index, unit='ms')

    # Train AR model
    model = AR(series)
    model_fit = model.fit(maxlag=10)
    ar_coeffs = np.append(ar_coeffs, model_fit.params.values[1:])
ar_coeffs
