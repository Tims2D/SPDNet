import os
import torch
from models import SPDNet, LSTM, RNN, GRU, BiLSTM,  ConvLSTM, ResLSTM, Conv2DLSTM, Times2D_Final, Times2D_Timmixer,iTransformer, Transformer, Informer,Times2D_3parts, TimeMixer, Crossformer, DLinear, Real_FITS, ModernTCN, HDMixer, SparseTSF, TimesNet, PatchTST, Times2D_3parts_FFT, GRU
            

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SPDNet': SPDNet,
            'LSTM': LSTM,
            'RNN': RNN,
            'GRU': GRU,
            'BiLSTM': BiLSTM,
            'Transformer': Transformer, 
            'ConvLSTM': ConvLSTM,
            'Conv2DLSTM': Conv2DLSTM,
            'ResLSTM': ResLSTM,
            'iTransformer': iTransformer,
            'Informer' : Informer,
            'Times2D_3parts' : Times2D_3parts,
            'TimeMixer': TimeMixer,
            'Crossformer': Crossformer,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'PatchTST': PatchTST, 
            'GRU' : GRU,
        }
        # 
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        model_name = self.args.model
        if model_name not in self.model_dict:
            raise ValueError(f"Model {model_name} not supported.")
        
        # Instantiate the model
        if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            # For ML models, no need to move to GPU, so bypass the `.to(self.device)`
            model = self.model_dict[model_name](self.args)
        else:
            # For DL models
            model = self.model_dict[model_name](self.args).to(self.device)
        
        return model
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Using GPU: {device}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device
    
    def _get_data(self):
        raise NotImplementedError
    
    def vali(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
