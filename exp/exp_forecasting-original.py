import math
import numpy as np
import torch
import torch.nn as nn
import os
import time
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

    # Add the missing `get_gpu_memory_usage` method
    def get_gpu_memory_usage(self):
        gpu_usage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
        return int(gpu_usage.strip())  # Convert string to integer (MB)

    def monitor_memory_usage(self, memory_usage_list, gpu_memory_usage_list, stop_event):
        peak_memory_usage = 0
        peak_gpu_memory_usage = 0
        while not stop_event.is_set():
            memory_usage = self.get_memory_usage()
            gpu_memory_usage = self.get_gpu_memory_usage()
            memory_usage_list.append(memory_usage)
            gpu_memory_usage_list.append(gpu_memory_usage)

            if gpu_memory_usage > peak_gpu_memory_usage:
                peak_gpu_memory_usage = gpu_memory_usage
            if memory_usage > peak_memory_usage:
                peak_memory_usage = memory_usage

            time.sleep(1)  
        return peak_memory_usage, peak_gpu_memory_usage
    
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

        
        time_list = []
        memory_usage_list = []
        gpu_memory_usage_list = []
        stop_event = threading.Event()
        process = psutil.Process() ###
        ram_before = process.memory_info().rss / 1024 ** 2 ###
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_memory_usage, args=(memory_usage_list, gpu_memory_usage_list, stop_event))
        monitor_thread.start()

        step_times = []  # List to store the time for each step**
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            RAM_Usage_list = []
            
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
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                                
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)
                elif self.task_name == 'Univariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    print('please ckeck the task name')
                    
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
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            step_time = time.time() - step_start_time  # Calculate step time**
            step_times.append(step_time)  # Store step time**
        
        
        ram_after = process.memory_info().rss / 1024 ** 2
            #print(f"RAM after {ram_after}")
        RAM_usage = ram_after - ram_before 
        RAM_Usage_list.append(RAM_usage)

        if epoch == self.args.train_epochs:
                RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)
                print(f"(original) RAM usage per epoch: {RAM_usage} MB")
        # Stop monitoring thread and capture peak values
        stop_event.set()
        monitor_thread.join()

        # Calculate and print the peak and average memory usage
        average_memory_usage = sum(memory_usage_list) / len(memory_usage_list)
        average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)
        average_step_time = sum(step_times) / len(step_times)  # Calculate average step time**
        print('_______________________________________Efficiency and Running Time_____________________________________')
        print(f"| {'Metric':<40} | {'Value':>20} |")
        print("--------------------------------------------------------------------------------------------------------")
        print(f"| {'Average time per step':<40} | {average_step_time:>20.6f} seconds |")
        print(f"| {'Average GPU memory usage':<40} | {average_gpu_memory_usage:>20.2f} MB |")
        print(f"| {'Average RAM usage':<40} | {average_memory_usage:>20.2f} MB |")
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

                outputs = self.model(batch_x)
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

                outputs = self.model(batch_x)

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

    