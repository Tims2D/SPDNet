import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, configs):
        """
        Initialize the custom dataset for load demand forecasting.

        Parameters are passed through a configuration object (configs).
        """
        # Extract configurations
        self.root_path = configs.root_path             # Path to the root directory where the dataset is stored
        self.data_path = configs.data_path             # File name or path to the specific dataset (CSV file)
        self.target = configs.target                   # Name of the target column that the model will predict
        self.features = configs.features               # Type of features : 'S' for single, 'M' for multi, 'MS' for multi with target included
        self.seq_len = configs.seq_len                 # Length of the input sequence (number of time steps the model will use as input)
        self.label_len = configs.label_len             # label sequence (number of time steps the model will predict during training)
        self.pred_len = configs.pred_len               # Length of the prediction sequence (number of future time steps the model will forecast)
        self.scale = configs.scale                     # Boolean flag indicating whether to scale the data (True for scaling, False otherwise)
        self.timeenc = configs.timeenc                 # Type of time encoding: 0 for manual encoding (month, day, etc.), 1 for learned encoding
        self.freq = configs.freq                       # Frequency of the data ('15min' for 15 minutes, '30min' for 30 minutes, 'h' for hourly)
        self.flag = configs.flag                       # Indicates the dataset split: 'train', 'val' (validation), or 'test'

        
        # Additional setup based on flag
        assert self.flag in ['train', 'test', 'val'], "flag must be 'train', 'test' or 'val'"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.flag]

        # Load data
        self.__read_data__()

    def __read_data__(self):
        # Load and preprocess data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        #imputer = SimpleImputer(strategy='mean')
        #df_raw = pd.DataFrame(imputer.fit_transform(df_raw), columns=df_raw.columns)
        
        #print("NaN in data_x:", np.isnan(df_raw).sum())
        # Determine the split points for training, validation, and testing
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Handle different feature types
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # Scale data if required
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        
        # Prepare time features based on the frequency
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute // 15 * 15) # adjust for 15min interval
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    
class ML_DataLoader(Dataset):
    def __init__(self, configs):
        """
        Initialize the dataset for traditional ML models like Random Forest, XGBoost, and LightGBM.

        Parameters:
        - configs: Configuration object containing dataset parameters.
        """
        self.root_path = configs.root_path
        self.data_path = configs.data_path
        self.target = configs.target
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = configs.scale

        self.scaler = StandardScaler()

        # Load, preprocess, and split data
        self.__read_data__()

    def __read_data__(self):
        """
        Load and preprocess the dataset. This includes loading the data,
        converting it to a supervised learning format, scaling if required,
        and splitting into training, validation, and test sets.
        """
        # Load the dataset
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.drop(columns=['date'])
        cols = list(df_raw.columns)
        cols.remove(self.target)
        df_raw = df_raw[cols + [self.target]]
        # Determine the split points for training, validation, and testing
        train_size = int(len(df_raw) * 0.7)
        test_size = int(len(df_raw) * 0.2)
        val_size = len(df_raw) - train_size - test_size
        border1s = [0, train_size - self.seq_len, len(df_raw) - test_size - self.seq_len]
        border2s = [train_size, train_size + val_size, len(df_raw)]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            scaled_data = self.scaler.transform(df_data.values)
        else:
            scaled_data = df_data.values
        # Prepare the data for supervised learning
        data = self.series_to_supervised(scaled_data, n_in=self.seq_len, n_out=self.pred_len)

        # Split into input and output
        data_x, data_y = data[:, :-self.pred_len], data[:, -self.pred_len:]

        # Split data into training, validation, and test sets
        self.X_train, self.y_train = data_x[:train_size], data_y[:train_size]
        self.X_val, self.y_val = data_x[train_size:train_size + val_size], data_y[train_size:train_size + val_size]
        self.X_test, self.y_test = data_x[train_size + val_size:], data_y[train_size + val_size:]

       
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
            """
            Frame a time series as a supervised learning dataset.
            """
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()

            # Input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

            # Forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

            # Put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names

            if dropnan:
                agg.dropna(inplace=True)

            return agg.values

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return len(self.X_train)

    def inverse_transform(self, data):
        """
        Inverse transform the scaled data to its original scale.
        """
        return self.scaler.inverse_transform(data)

