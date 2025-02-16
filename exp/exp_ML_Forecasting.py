import os
import numpy as np
from torch.utils.data import DataLoader
from exp.exp_basic import Exp_Basic
from data_provider.data_loader import ML_DataLoader
from models.RandomForest import RandomForest  # Import the RandomForest model

class Exp_ML_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_ML_Forecasting, self).__init__(args)
        self.model = self._build_model()

    def _build_model(self):
        # Initialize the RandomForest model
        model = RandomForest(self.args)
        return model

    def _get_data(self, flag):
        # Load data using ML_DataLoader
        data_set = ML_DataLoader(self.args)
        data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=(flag == 'train'))
        return data_set, data_loader

    def train(self, setting):
        # Load training data
        train_data, train_loader = self._get_data(flag='train')

        # Prepare training input and output
        X_train = np.array([x for x, _ in train_loader])
        y_train = np.array([y for _, y in train_loader])

        # Train the model
        self.model.fit(X_train, y_train)

        # Save the trained model
        model_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save_model(os.path.join(model_path, 'random_forest_model.pkl'))

        # Evaluate on validation data
        vali_data, vali_loader = self._get_data(flag='val')
        vali_loss = self.vali(vali_data, vali_loader)

        # Evaluate on test data
        test_data, test_loader = self._get_data(flag='test')
        test_loss = self.vali(test_data, test_loader)

        print(f"Validation Loss: {vali_loss:.6f}, Test Loss: {test_loss:.6f}")

        return self.model

    def vali(self, vali_data, vali_loader):
        # Validation logic
        X_val = np.array([x for x, _ in vali_loader])
        y_val = np.array([y for _, y in vali_loader])

        # Make predictions
        y_pred = self.model.predict(X_val)

        # Calculate and return the loss (MSE)
        mse = np.mean((y_val - y_pred) ** 2)
        return mse

    def test(self, setting, load=False):
        # Load test data
        test_data, test_loader = self._get_data(flag='test')

        if load:
            model_path = os.path.join(self.args.checkpoints, setting, 'random_forest_model.pkl')
            self.model.load_model(model_path)

        # Prepare test input and output
        X_test = np.array([x for x, _ in test_loader])
        y_test = np.array([y for _, y in test_loader])

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate and print metrics
        mse = np.mean((y_test - y_pred) ** 2)
        print(f"Test MSE: {mse:.6f}")

        # Save predictions and true values
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred.npy', y_pred)
        np.save(folder_path + 'true.npy', y_test)

        return mse
