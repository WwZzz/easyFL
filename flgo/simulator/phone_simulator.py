import copy
import warnings

from flgo.simulator.base import BasicSimulator
import os
import zipfile
try:
    import pandas as pd
except:
    pd = None
import urllib
import numpy as np
import re
import random
"""
************************* Availability ***************************
We construct the client availability according to the public real-world dataset "Users Active 
Time Prediction". This is data dealing with mobile app usage of the customers, where an app 
has some personal information and online active timing of the customers. Whenever the 
customers login in-app and view anything, the app server gets pings from their mobile phone 
indicating that they are using the app.  More information about this dateset is in
https://www.kaggle.com/datasets/bhuvanchennoju/mobile-usage-time-prediction .
Before using this simulator, you should manually download and unzip the original file in kaggle to
ensure the  existence of file `benchmark/RAW_DATA/USER_ACTIVE_TIME/pings.csv`.
"""
class Simulator(BasicSimulator):
    def __init__(self, objects = [], option = None):
        super().__init__(objects)
        self.option = option
        self.num_clients = len(self.clients)
        self.customer_map = {}
        # availability
        self.init_availability(os.path.join('benchmark', 'RAW_DATA', 'USER_ACTIVE_TIME'))
        # computing resources
        self.init_computing_resource(os.path.join('benchmark', 'RAW_DATA', 'BODT'))
        # upload & download speed
        self.init_network(os.path.join('benchmark', 'RAW_DATA', 'OoklaSpeedtest'))

    def init_availability(self, rawdata_path):
        zipfile_path = os.path.join(rawdata_path,'archive.zip')
        data_path = os.path.join(rawdata_path,'pings.csv')
        if not os.path.exists(data_path) and os.path.exists(zipfile_path):
            f = zipfile.ZipFile(zipfile_path, 'r')
            for file in f.namelist():
                f.extract(file, rawdata_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError("Please download the original dataset in https://www.kaggle.com/datasets/bhuvanchennoju/mobile-usage-time-prediction , and move it into `benchmark/RAW_DATA/USER_ACTIVE_TIME/pings.csv`")
        customers_info = pd.read_csv(os.path.join(rawdata_path, "customers.csv"))
        customers_info = customers_info.drop_duplicates(['id'], ignore_index=True)
        customers_availability = pd.read_csv(os.path.join(rawdata_path, 'pings.csv'))
        customers_availability['timestamp'] = customers_availability['timestamp'] - customers_availability['timestamp'][0]
        customers_availability_by_time = customers_availability.groupby('timestamp')['id'].apply(list)
        customers_availability_by_id = customers_availability.groupby('id')['timestamp'].apply(list)
        customers_availability_by_id = customers_availability_by_id.to_frame(name='timestamps')
        customers_availability_by_id = customers_availability_by_id.reset_index()
        if self.option['availability'] == 'IDL' or (self.option['availability']=='HIGH' and self.num_clients<len(customers_availability_by_id)) :
            customers_availability_by_id['num_stamps'] = customers_availability_by_id.apply(lambda x: len(x['timestamps']), axis=1)
            customers_availability_by_id = customers_availability_by_id.sort_values(by='num_stamps', ascending=False)
            customers_availability_by_id = customers_availability_by_id['id'].to_list()[:self.num_clients]
            self.customer_map = {cid: customers_availability_by_id[cid] for cid in range(self.num_clients)}
        elif self.option['availability'] == 'LOW' and self.num_clients<len(customers_availability_by_id):
            customers_availability_by_id['num_stamps'] = customers_availability_by_id.apply(lambda x: len(x['timestamps']), axis=1)
            customers_availability_by_id = customers_availability_by_id.sort_values(by='num_stamps', ascending=True)
            customers = customers_availability_by_id['id'].to_list()[:self.num_clients]
            self.customer_map = {cid: customers[cid] for cid in range(self.num_clients)}
        elif self.option['availability']=='RANDOM':
            replacement = True  if self.num_clients>len(customers_availability_by_id) else False
            customers = customers_availability_by_id.sample(n=self.num_clients, replace=replacement)['id'].to_list()
            self.customer_map = {cid: customers[cid] for cid in range(self.num_clients)}
        else:
            raise NotImplementedError("Availability {} has been not implemented.".format(self.option['availability']))
        self.availability_table = customers_availability_by_time
        # def visualize(a, b, k=10000):
        #     while a%15!=0: a=a+1
        #     import matplotlib.pyplot as plt
        #     ratio = max( (b-a)//k, 1)
        #     plt.figure(figsize=(ratio*2.56, 2.56))
        #     state = [0 for _ in range(len(customers_info))]
        #     time_start = [0 for _ in range(len(customers_info))]
        #     time_end = [0 for _ in range(len(customers_info))]
        #     for tid in range(a, b, 15):
        #         try:
        #             cids = customers_availability_by_time[tid]
        #         except:
        #             continue
        #         y = customers_info.loc[customers_info['id'].isin(cids)].index.to_list()
        #         for cid in range(len(customers_info)):
        #             if cid in y:
        #                 if state[cid]==0:
        #                     state[cid] = 1
        #                     time_start[cid] = tid
        #             else:
        #                 if state[cid]==1:
        #                     state[cid]=0
        #                     time_end[cid] = tid
        #                     plt.plot([time_start[cid], time_end[cid]], [cid, cid], c='g')
        #     plt.show()
        return

    def init_computing_resource(self, rawdata_path):
        # u_k = f(dev_k, model_k)
        # TimePerBatch_k =  u_k * batch_size * model_size
        # TimeLocalTraining_k = TimePerBatch * working_amount
        # k_dev should be initialized here
        github_url = "https://github.com/UbiquitousLearning/Benchmark-On-Device-Training/tree/master/tools/train/benchmark"
        device_names = ["MEIZU_Freq_1708800", "XIAOMI","MEIZU_Core_f0"]
        models = ["Lenet", "Squeezenet", "Mobilenet", "Alexnet"]
        model_sizes = {
            "Lenet": 3.2,
            "Squeezenet": 441.2,
            "Mobilenet": 3400,
            "Alexnet": 61000,
        }
        file_tmp = 'time_table_{}.csv'
        if not os.path.exists(os.path.join(rawdata_path, 'XIAOMI')):
            for dev in device_names:
                device_path = os.path.join(rawdata_path, dev)
                if not os.path.exists(device_path):
                    os.mkdir(device_path)
                for model in models:
                    model_url = os.path.join(github_url, dev, 'processed_data', model, file_tmp.format(model))
                    urllib.request.urlretrieve(model_url, device_path)
        us = []
        for dev in device_names:
            device_path = os.path.join(rawdata_path, dev)
            u_models = []
            for model in models:
                filename = 'time_table_{}.csv'.format(model)
                filepath = os.path.join(device_path, filename)
                if os.path.exists(filepath):
                    data = pd.read_csv(filepath)
                    batchsizes = data['batchsize'].to_list()
                    time_per_batch = data['expr_train'].to_list()
                    u_current_model = []
                    for bs, btime in zip(batchsizes, time_per_batch):
                        u_current_model.append(1.0*btime/model_sizes[model]/bs)
                    u_current_model = np.array(u_current_model).mean()
                    u_models.append(u_current_model)
            if len(u_models)>0: us.append(np.array(u_models).mean())
        us = np.array(us)
        if "LCLN" in self.option['responsiveness']:
            flag = self.option['responsiveness'].find('LCLN-')
            tmp  = self.option['responsiveness']
            if not tmp.endswith('_'): tmp = tmp + '_'
            std = float(tmp[flag+5: tmp.find('_', flag)])
            u_mean = us.mean()
            mu = np.log(u_mean) - std**2/2.0
            self.client_us = np.random.lognormal(mu, std, self.num_clients).tolist()
        else:
            ulow = us.min()
            uhigh = us.max()
            self.client_us = np.random.uniform(low = ulow, high = uhigh, size=self.num_clients).tolist()

    def init_network(self, rawdata_path):
        # load Ookla dataset
        zipfile_path = os.path.join(rawdata_path, 'archive.zip')
        data_path = os.path.join(rawdata_path, 'fixed_year_2020_quarter_01.csv')
        if not os.path.exists(data_path) and os.path.exists(zipfile_path):
            f = zipfile.ZipFile(zipfile_path, 'r')
            for file in f.namelist():
                f.extract(file, rawdata_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError("Please download the original dataset in https://www.kaggle.com/datasets/dimitrisangelide/speedtest-data-by-ookla , and move it into `benchmark/RAW_DATA/OoklaSpeedtest/fixed_year_2020_quarter_01.csv`")

        data = pd.read_csv(data_path) # 获取colums
        # 数据集分为fixed和mobile两类，共22个csv文件，按年份和季度记录数据
        # 用到数据集中的Avg. Avg U Kbps（上传速度）和Avg. Avg D Kbps（下载速度）
        # 思路是把fixed和mobile的分开来然后合并所有的年份和季度数据，
        # 根据国家name去重（国家看作client），再将两个速度取均值
        Mobile_df = pd.DataFrame([], columns=data.columns)
        Broadband_df = pd.DataFrame([], columns=data.columns)
        for dirname, _, filenames in os.walk(rawdata_path):
            for filename in filenames:
                if filename.endswith('.zip'): continue
                meta_info = filename.split('/')[-1]
                data = pd.read_csv(dirname + '/' + filename, thousands=r',').convert_dtypes()
                if set(data.columns).intersection(set({'Number of Record': 'Number of Records'}.keys())):
                    data.rename(columns={'Number of Record': 'Number of Records'}, inplace=True)
                # data = self.col_name_corrections(data, {'Number of Record': 'Number of Records'})
                data['Year'] = np.int64(re.search('year_(.*)_quarter', meta_info).group(1))
                data['Quarter'] = np.int64(re.search('quarter_(.*).csv', meta_info).group(1))
                if 'mobile' in meta_info:
                    Mobile_df = pd.concat([Mobile_df, data])
                else:
                    Broadband_df = pd.concat([Broadband_df, data])

        Mobile_df = Mobile_df.astype({'Year': np.int64, 'Quarter': np.int64})
        Broadband_df = Broadband_df.astype({'Year': np.int64, 'Quarter': np.int64})
        Mobile_df.sort_values(by=['Year', 'Quarter'], ascending=[True, True], inplace=True)
        Broadband_df.sort_values(by=['Year', 'Quarter'], ascending=[True, True], inplace=True)
        unique_countries_broadband = Broadband_df.groupby('Name').agg({'Number of Records': 'sum',
                                                                       'Devices':'sum',
                                                                       'Tests':'sum',
                                                                       'Avg. Avg U Kbps': np.mean,
                                                                       'Avg. Avg D Kbps': np.mean,
                                                                       'Avg Lat Ms': np.mean,
                                                                       'Avg. Pop2005': np.mean,
                                                                       'Rank Upload': np.mean,
                                                                       'Rank Download': np.mean,
                                                                       'Rank Latency': np.mean,
                                                                       'Year': np.mean,
                                                                       'Quarter': np.mean,
                                                                       })
        unique_countries_mobile = Mobile_df.groupby('Name').agg({'Number of Records': 'sum',
                                                                       'Devices': 'sum',
                                                                       'Tests': 'sum',
                                                                       'Avg. Avg U Kbps': np.mean,
                                                                       'Avg. Avg D Kbps': np.mean,
                                                                       'Avg Lat Ms': np.mean,
                                                                       'Avg. Pop2005': np.mean,
                                                                       'Rank Upload': np.mean,
                                                                       'Rank Download': np.mean,
                                                                       'Rank Latency': np.mean,
                                                                       'Year': np.mean,
                                                                       'Quarter': np.mean,
                                                                       })
        # 分fixed和mobile根据上传，下载速度计算时间
        # 1 kbps = 10^3 bit/s
        broadband_id_list = [ i for i in range(unique_countries_broadband.shape[0])]
        broadband_upload_speed_list =  unique_countries_broadband['Avg. Avg U Kbps'].to_list()
        broadband_download_speed_list = unique_countries_broadband['Avg. Avg D Kbps'].to_list()
        # KB/s
        broadband_dev_list = [(u*125,d*125) for u,d in zip(broadband_upload_speed_list, broadband_download_speed_list)]
        mobile_id_list = [i for i in range(unique_countries_mobile.shape[0])]
        mobile_upload_speed_list = unique_countries_mobile['Avg. Avg U Kbps'].to_list()
        mobile_download_speed_list = unique_countries_mobile['Avg. Avg D Kbps'].to_list()
        mobile_dev_list = [(u*125,d*125) for u,d in zip(mobile_upload_speed_list, mobile_download_speed_list)]
        net_dev_list = broadband_dev_list + mobile_dev_list
        if 'NTH' in self.option['responsiveness'] or 'NTL' in self.option['responsiveness']:
            reverse = True if 'NTH' in self.option['responsiveness'] else False
            if self.num_clients <= len(net_dev_list):
                net_dev_list.sort(key = lambda x:x[0]+x[1], reverse=reverse)
                self.net_speeds = net_dev_list[:self.num_clients]
            else:
                self.net_speeds = net_dev_list * (self.num_clients/len(net_dev_list))
                num_rest = self.num_clients % len(net_dev_list)
                net_dev_list.sort(key=lambda x: x[0] + x[1], reverse=reverse)
                self.net_speeds = self.net_speeds + net_dev_list[:num_rest]
        else:
            replacement = True if self.num_clients > len(net_dev_list) else False
            net_ids = np.random.choice(list(range(len(net_dev_list))), size=self.num_clients, replace=replacement)
            self.net_speeds = [net_dev_list[nid] for nid in net_ids]

    def update_client_availability(self):
        t = self.gv.clock.current_time
        t = t%self.availability_table.index[-1]
        aid = t-t%15
        try:
            available_customers = self.availability_table[aid]
        except:
            available_customers = []
        pa, pua = [], []
        for cid in self.all_clients:
            pai = 1.0 if self.customer_map[cid] in available_customers else 0.0
            pa.append(pai)
            pua.append(1.0-pai)
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)

    def update_client_responsiveness(self, client_ids, *args, **kwargs):
        # calculate time of uploading and downloading model
        latency = []
        up_pkg_sizes = self.get_variable(client_ids, '__upload_package_size')
        down_pkg_sizes = self.get_variable(client_ids, '__download_package_size')
        for cid, upsize,down_size in zip(client_ids, up_pkg_sizes, down_pkg_sizes):
            latency.append(int(1.0*upsize/self.net_speeds[cid][0] + 1.0*down_size/self.net_speeds[cid][1]))
        # calculate time of local_movielens_recommendation computing
        tlcs = []
        model_sizes =  self.get_variable(client_ids, '__model_size')
        working_amounts = self.get_variable(client_ids, 'working_amount')
        for cid, model_size, wa in zip(client_ids, model_sizes, working_amounts):
            uk = self.client_us[cid]
            tlc_k = uk*model_size/1000.0*self.clients[cid].batch_size*wa/1000.0
            tlcs.append(tlc_k)
        for idx, _ in enumerate(client_ids):
            latency[idx] = int(latency[idx] + tlcs[idx])
        self.set_variable(client_ids, 'latency', latency)

    def update_client_completeness(self, client_ids, *args, **kwargs):
        pass