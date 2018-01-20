import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.ar_model import AR

# Fetching data
# Train

pt_array = ["Ainput", "Binput"]
p_train_beh_array = ["train_beh_1", "train_beh_2", "train_beh_3", "train_beh_4",
                     "train_beh_5", "train_beh_6", "train_beh_7", "train_beh_8"]
p_train_sub_array = ["train_sub_1", "train_sub_2", "train_sub_3", "train_sub_4", "train_sub_5", "train_sub_6", "train_sub_7", "train_sub_8", "train_sub_9", "train_sub_10", "train_sub_1", "train_sub_12", "train_sub_13",
                     "train_sub_14", "train_sub_15", "train_sub_16", "train_sub_17", "train_sub_18", "train_sub_19", "train_sub_20", "train_sub_21", "train_sub_22", "train_sub_23", "train_sub_24", "train_sub_25"]
p_train_fn_array = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
names = ['time', 'X_acc', 'Y_acc', 'Z_acc', 'X_gyro', 'Y_gyro', 'Z_gyro']

pt_in = pt_array[1]
pt_out = "Posture" if pt_in == "Binput" else "Trunk"
new_ind_data = [[]]
new_sub_data = [[]]
new_main_data = [[]]
sub_num = 0
beh_num = 0


# Train data: extract features and make new data
for sub in p_train_sub_array:
    beh_num = 0
    sub_num += 1
    for beh in p_train_beh_array:
        beh_num += 1
        for fn in p_train_fn_array:
            file_path = "data" + \
                pt_in + "\\" + sub + "\\" + \
                beh + "\\" + sub + "_" + beh + fn + ".csv"
            dataset = pd.read_csv(file_path, index_col='time', names=names)
            # Getting stat features
            new_row_data = dataset.describe().T[['mean', 'std', 'min', 'max']].values.flatten()

            # Dong SMA on data for AR model use (Removes noise for some degree)
            dataset['roll_mean_X_acc'] = dataset['X_acc'].rolling(window=50).mean()
            dataset['roll_mean_Y_acc'] = dataset['Y_acc'].rolling(window=50).mean()
            dataset['roll_mean_Z_acc'] = dataset['Z_acc'].rolling(window=50).mean()
            dataset['roll_mean_X_gyro'] = dataset['X_gyro'].rolling(window=50).mean()
            dataset['roll_mean_Y_gyro'] = dataset['Y_gyro'].rolling(window=50).mean()
            dataset['roll_mean_Z_gyro'] = dataset['Z_gyro'].rolling(window=50).mean()
            data = dataset.values

            # AR model coeff
            ar_coeffs = []
            time_index = pd.to_datetime(range(len(data[49:, 0])),  unit='ms')
            for i in range(6, 9):
                series = pd.Series(data[49:, i], index=time_index)
                # Train AR model
                model = AR(series)
                model_fit = model.fit(maxlag=10)
                ar_coeffs = np.append(ar_coeffs, model_fit.params.values[1:])
            new_row_data = np.append(new_row_data, ar_coeffs)

            """
            # Getting Slope
            slopes = []
            time_axis = range(len(data[:, 0]))
            for i in range(0, 3):
                new_slope = stats.linregress(time_axis, data[:, i]).slope
                slopes = np.append(slopes, new_slope)
            new_row_data = np.append(new_row_data, slopes)
            """
            new_row_data = np.append(new_row_data, [sub_num, beh_num])

            if(fn == "1"):
                new_ind_data = new_row_data
            else:
                new_ind_data = np.vstack((new_ind_data, new_row_data))
        # Save data according to sub + beh
        np.savetxt("data" + pt_out + "\\train_data\\" +
                   pt_out + "_train_feature_ex\\" + sub + "_" + beh + ".csv", new_ind_data, delimiter=",")
        if(beh == "train_beh_1"):
            new_sub_data = new_ind_data
        else:
            new_sub_data = np.vstack((new_sub_data, new_ind_data))
    # save data according to sub
    np.savetxt("data" + pt_out + "\\train_data\\" +
               pt_out + "_train_feature_sub\\" + sub + ".csv", new_sub_data, delimiter=",")
    if(sub == "train_sub_1"):
        new_main_data = new_sub_data
    else:
        new_main_data = np.vstack((new_main_data, new_sub_data))
# save whole data
new_main_data
np.savetxt("data\\train_input.csv", new_main_data, delimiter=",")


# Test:
p_test_beh_array = ["test_beh_1", "test_beh_2", "test_beh_3", "test_beh_4",
                    "test_beh_5", "test_beh_6", "test_beh_7", "test_beh_8"]
p_test_sub_array = ["test_sub_1", "test_sub_2", "test_sub_3", "test_sub_4", "test_sub_5"]
p_test_fn_array = ["1", "2", "3", "4", "5"]
names = ['time', 'X_acc', 'Y_acc', 'Z_acc', 'X_gyro', 'Y_gyro', 'Z_gyro']

new_ind_data = [[]]
new_sub_data = [[]]
new_main_data = [[]]
sub_num = 0
beh_num = 0
# Test data: add features and make new data
for sub in p_test_sub_array:
    sub_num += 1
    beh_num = 0
    for beh in p_test_beh_array:
        beh_num += 1
        for fn in p_test_fn_array:
            file_path = "data" + \
                pt_in + "\\" + sub + "\\" + \
                beh + "\\" + sub + "_" + beh + fn + ".csv"
            dataset = pd.read_csv(file_path, index_col='time', names=names)
            new_row_data = dataset.describe().T[['mean', 'std', 'min', 'max']].values.flatten()
            dataset['roll_mean_X_acc'] = dataset['X_acc'].rolling(window=50).mean()
            dataset['roll_mean_Y_acc'] = dataset['Y_acc'].rolling(window=50).mean()
            dataset['roll_mean_Z_acc'] = dataset['Z_acc'].rolling(window=50).mean()
            dataset['roll_mean_X_gyro'] = dataset['X_gyro'].rolling(window=50).mean()
            dataset['roll_mean_Y_gyro'] = dataset['Y_gyro'].rolling(window=50).mean()
            dataset['roll_mean_Z_gyro'] = dataset['Z_gyro'].rolling(window=50).mean()
            data = dataset.values

            # AR model coeff
            ar_coeffs = []
            time_index = pd.to_datetime(range(len(data[49:, 0])),  unit='ms')
            for i in range(6, 9):
                series = pd.Series(data[49:, i], index=time_index)
                # Train AR model
                model = AR(series)
                model_fit = model.fit(maxlag=10)
                ar_coeffs = np.append(ar_coeffs, model_fit.params.values[1:])
            new_row_data = np.append(new_row_data, ar_coeffs)

            """
            # Getting Slope
            time_axis = range(len(data[:, 0]))
            slopes = []

            for i in range(0, 3):
                new_slope = stats.linregress(time_axis, data[:, i]).slope
                slopes = np.append(slopes, new_slope)

            new_row_data = np.append(new_row_data, slopes)
            """
            if beh_num < 5:
                test_beh_num = 1
            elif beh_num < 7:
                test_beh_num = 2
            else:
                test_beh_num = 3
            new_row_data = np.append(new_row_data, [sub_num, test_beh_num])

            if(fn == "1"):
                new_ind_data = new_row_data
            else:
                new_ind_data = np.vstack((new_ind_data, new_row_data))
        # save data according to sub + beh
        np.savetxt("data" + pt_out + "\\test_data\\" +
                   pt_out + "_test_feature_ex\\" + sub + "_" + beh + ".csv", new_ind_data, delimiter=",")
        if(beh == "test_beh_1"):
            new_sub_data = new_ind_data
        else:
            new_sub_data = np.vstack((new_sub_data, new_ind_data))
    # save data according to sub
    np.savetxt("data" + pt_out + "\\test_data\\" +
               pt_out + "_test_feature_sub\\" + sub + ".csv", new_sub_data, delimiter=",")
    if(sub == "test_sub_1"):
        new_main_data = new_sub_data
    else:
        new_main_data = np.vstack((new_main_data, new_sub_data))
# save whole data
new_main_data
np.savetxt("data\\test_input.csv", new_main_data, delimiter=",")
