import math
import numpy as np
import torch
import torch.nn as nn
import os
import time
import ptflops ###
import warnings
import matplotlib.pyplot as plt
from torch import optim
import psutil
import threading
from torch.optim import lr_scheduler
import torch.nn.functional as F
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
warnings.filterwarnings('ignore')


class Exp_demand_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_demand_Forecasting, self).__init__(args)
        self.task_name =args.task_name
    def _build_model(self):
       
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

    def get_gpu_memory_usage(self):
        
        try:
        
            gpu_usage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
            return int(gpu_usage.strip())  # Convert string to integer (MB)
        
        except Exception as e:
            
            return 0  # Or any other value indicating that the usage is not available
    
    
    def monitor_memory_usage(self, memory_usage_list, gpu_memory_usage_list, stop_event):
        while not stop_event.is_set():
            memory_usage = self.get_memory_usage()
            gpu_memory_usage = self.get_gpu_memory_usage()
            memory_usage_list.append(memory_usage)
            gpu_memory_usage_list.append(gpu_memory_usage)
            
            
            time.sleep(1)  # Adjust the sleep time as needed
            
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

       # RAM_Usage meaurement
        process = psutil.Process() ###
        ram_before = process.memory_info().rss / 1024 ** 2 ###
        print(f"RAM before {ram_before}")
        
        time_list = []
        RAM_Usage_list = []
        memory_usage_list = []
        gpu_memory_usage_list = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self.monitor_memory_usage, args=(memory_usage_list, gpu_memory_usage_list, stop_event))
        monitor_thread.start()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                step_start_time = time.time()  # Start timing for this step
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #print('self.args.model =', self.args.model)
                
                if self.args.model in ['RNN', 'LSTM', 'BiLSTM', 'ResLSTM', 'Real_FITS', 'ModernTCN', 'SparseTSF']:
                    #print('we are working in if')
                    outputs = self.model(batch_x)
                
                elif self.args.model in ['HDMixer']:
                    outputs,PaEN_Loss = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    #print('we are working in else')
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                                
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)   # batch_y =[B, Pred_len, 1]
                     #print('we are working')
                elif self.task_name == 'Univariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    print('please ckeck the task name')

                #print('outputs shape =', outputs.shape)    # [B, Pred_len, 1]  
                #print('batch_y shape =', batch_y.shape)    # [B, Pred_len, 1]
                
                if self.args.model in ['HDMixer']:
                    mseloss = criterion(outputs, batch_y)
                    loss = mseloss+PaEN_Loss
                    train_loss.append(mseloss.item())
                    
                else:   
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.cuda.empty_cache()  # Clear unused memory
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                ### training time measurement
                
                t1 = (time.time() - epoch_time)
                time_list.append(t1)
                average_time = sum(time_list) / len(time_list)
                
                ### RAM_Usage

                ram_after = process.memory_info().rss / 1024 ** 2

                RAM_usage = ram_after - ram_before 

                RAM_Usage_list.append(RAM_usage)

                RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)
                    
                ### RAM_Usage Using Thread
                    
                stop_event.set()
                monitor_thread.join()

                average_memory_usage = sum(memory_usage_list) / len(memory_usage_list)
                average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)

                ###
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        
        ### training time measurement
                
        t1 = (time.time() - epoch_time)
        time_list.append(t1)
        average_time = sum(time_list) / len(time_list)


        ### RAM_Usage

        ram_after = process.memory_info().rss / 1024 ** 2

        RAM_usage = ram_after - ram_before

        RAM_Usage_list.append(RAM_usage)

        RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)

            ### RAM_Usage Using Thread
            
        print(f"epoch is {epoch}")

        if epoch == (self.args.train_epochs - 1):
                    
                stop_event.set()
                monitor_thread.join()

        average_memory_usage = sum(memory_usage_list) / len(memory_usage_list)
        average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)

        ### Print

        print('_______________________________________Efficiency and Running Time_____________________________________')
        print(f"| {'Metric':<40} | {'Value':>20} |")
        print("--------------------------------------------------------------------------------------------------------")

        print(f"| {'Average training time per epoch':<40} | {average_time:>20.4f} seconds |")
                
        print(f"| {'RAM before':<40} | {ram_before:>20.2f} MB |")
        print(f"| {'RAM after':<40} | {ram_after:>20.2f} MB |")
            
        print(f"| {'RAM usage (After -Before) per epoch':<40} | {RAM_usage:>20.2f} MB |")
            
        print(f"| {'Average memory usage':<40} | {average_memory_usage:>20.2f} MB |")
        print(f"| {'Average GPU memory usage':<40} | {average_gpu_memory_usage:>20.2f} MB |")
        print()
        print("--------------------------------------------------------------------------------------------------------")
        

        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # global_encoder - decoder

                if self.args.model in ['RNN', 'LSTM', 'BiLSTM', 'ResLSTM', 'Real_FITS', 'ModernTCN', 'SparseTSF']:                    
                    outputs = self.model(batch_x)
                    
                elif self.args.model in ['HDMixer']:
                    outputs,PaEN_Loss = self.model(batch_x)
                
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                #outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)
                elif self.task_name == 'Univariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    print('please ckeck the task name')
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # global_encoder - decode

                if self.args.model in ['RNN', 'LSTM', 'BiLSTM', 'ResLSTM','Real_FITS', 'ModernTCN', 'SparseTSF']:
                    
                    outputs = self.model(batch_x)
                elif self.args.model in ['HDMixer']:
                    outputs,PaEN_Loss = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)
                        
                elif self.task_name == 'Univariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                else:
                    print('please ckeck the task name')
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs  #   [B, T, N]
                true = batch_y  # 
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # global_encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'PDF' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'PDF' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    