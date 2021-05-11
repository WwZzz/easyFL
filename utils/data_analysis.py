from utils.tools import *
import matplotlib.pyplot as plt
import json
import numpy as np

def table_cifar(filters={'lr':'0.1'}, path='../task/cifar10/record/'):
    heads = [
        'fedavg',
        'qfedavg_q0.1',
        'qfedavg_q0.2',
        'qfedavg_q1.0',
        'qfedavg_q2.0',
        'qfedavg_q5.0',
        'fedfa',
        'fedmgda+_epsilon0.01',
        'fedmgda+_epsilon0.05',
        'fedmgda+_epsilon0.1',
        'fedmgda+_epsilon0.5',
        'fedmgda+_epsilon1',
        'fedfv_alpha0.1_tau0',
        'fedfv_alpha0.2_tau0',
        'fedfv_alpha0.5_tau0',
        'fedfv_alpha0.1_tau1',
        'fedfv_alpha0.1_tau3',
        'fedfv_alpha0.1_tau10',
    ]
    K = 20
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.json')]
    for key in filters:
        files = [f for f in files if key+filters[key] in f]
    for head in heads:
        fnames = [f for f in files if head in f]
        means = []
        vars = []
        worstKs = []
        bestKs = []
        for f in fnames:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            means.append(cdata['mean_curve'][-1])
            vars.append(cdata['var_curve'][-1])
            accdist = cdata['acc_dist']
            accdist.sort()
            ws=int(len(accdist)*K*0.01)
            worstKs.append(np.mean(accdist[:ws]))
            bestKs.append(np.mean(accdist[-ws:]))
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(means), np.std(means)), end=' ')
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(vars), np.std(vars)), end=' ')
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(worstKs), np.std(worstKs)), end=' ')
        print("& {:.2f}$\\pm${:.2f}\\\\".format(np.mean(bestKs), np.std(bestKs)))
    return 0

def table_fmnist(filters={'lr':'0.1'}, path=''):
    heads = [
        'fedavg',
        'afl_learning_rate_lambda0.01',
        'afl_learning_rate_lambda0.1',
        'afl_learning_rate_lambda0.5',
        'qfedavg_q5',
        'qfedavg_q15',
        'fedfa',
        'fedmgda+_epsilon0.05',
        'fedmgda+_epsilon0.1',
        'fedmgda+_epsilon1',
        'fedfv_alpha0_',
        'fedfv_alpha0.334',
        'fedfv_alpha0.667',
    ]
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.json')]
    for key in filters:
        files = [f for f in files if key + filters[key] in f]
    for head in heads:
        fnames = [f for f in files if head in f]
        means = []
        vars = []
        uT=[]
        uP=[]
        uS=[]
        for f in fnames:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            means.append(cdata['mean_curve'][-1])
            vars.append(cdata['var_curve'][-1])
            client_accs=cdata['client_accs']
            uT.append(client_accs['T-shirt'][-1])
            uP.append(client_accs['pullover'][-1])
            uS.append(client_accs['shirt'][-1])
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(uS), np.std(uS)), end=' ')
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(uP), np.std(uP)), end=' ')
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(uT), np.std(uT)), end=' ')
        print("& {:.2f}$\\pm${:.2f}".format(np.mean(means), np.std(means)), end=' ')
        print("& {:.2f}$\\pm${:.2f}\\\\".format(np.mean(vars), np.std(vars)))

def table_order(dataset):
    heads = [
        'fedfv_',
        'fedfv_random',
        'fedfv_reverse',
    ]
    path = '../task/'+dataset+'/record/'
    files = os.listdir(path)
    for head in heads:
        fnames = [f for f in files if head in f and 'p0.2' in f]
        vars = []
        for f in fnames:
            file_path = os.path.join(path, f)
            with open(file_path, encoding='utf-8-sig', errors='ignore') as inf:
                cdata = json.load(inf, strict=False)
            vars.append(cdata['var_curve'][-1])
        print("{:.2f}$\\pm${:.2f} &".format(np.mean(vars),np.std(vars)), end=' ')

def data_for_plot(filters={'lr':'0.1'}, path='.', methods=[]):
    outdict={}
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.json')]
    for key in filters:
        files = [f for f in files if key + filters[key] in f]
    for method in methods:
        fnames = [f for f in files if method in f]
        means = []
        vars = []
        for f in fnames:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            means.append(cdata['mean_curve'])
            vars.append(cdata['var_curve'])
        outdict[method] = {}
        # averaging results of each method with different random seeds
        outdict[method]['ave']=np.mean(means,axis=0)
        outdict[method]['var'] = np.mean(vars, axis=0)
    return outdict


def draw_data(dataset, methods=[] ,labels=[], curve='ave'):
    colors = ['m','g','y','c','b','r','seagreen','darkblue','darkgreen','darkred']
    data = data_for_plot({'lr':'0.01'}, '../task/' + dataset + '/record/', methods)
    if dataset=='cifar10':
        R = 2000
        x = np.linspace(0, R, 40)
    else:
        R = 1000
        x = np.linspace(0,R,20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('communication round',fontsize=18)
    ylabel= 'mean of test accuracy' if curve== 'ave' else 'variance of test accuracy'
    plt.ylabel(ylabel,fontsize=18)
    plt.xlim(0, R)
    plt.ylim(5, 100)
    for i,key in enumerate(data.keys()):
        plt.plot(x, np.asarray(data[key][curve][:-2]), colors[i], label=labels[i],linewidth='2')
    plt.legend(loc='higher right',fontsize=15)
    res=plt.gcf()
    # res.savefig('fig2d.eps', format='eps',dpi=10000)
    plt.show()

if __name__ == '__main__':
    dataset='cifar10'
    methods = [
        'fedavg',
        'qfedavg_q1.0',
        'fedfa_beta0.5_gamma0.5_momentum0.5',
        'fedmgda+_epsilon0.1',
        'fedfv_alpha0.1_tau0',
        'fedfv_alpha0.2_tau10',
    ]
    labels = [
        'fedavg',
        'qfedavg q=1.0',
        'fedfa α=β=0.5',
        'fedmgda+ ε=0.1',
        'fedfv α=0.1 τ=0',
        'fedfv α=0.2 τ=10',
    ]
    draw_data(dataset, methods=methods, labels=labels, curve='ave')
    draw_data(dataset, methods=methods, labels=labels, curve='var')
