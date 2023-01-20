import time
from train_test import train,test
from IMC_dataset import IMC_dataset
import numpy as np
from gat import GatNet
import torch
import torch.nn as nn
import dgl
import random
import os
import scipy.io as sio


def saveToFile(url,also_print=True):
    with open(url,"w",encoding="utf8") as f:
        print(str(time.localtime()),file=f)
    def write(*args):
        with open(url,"a+",encoding="utf8") as f:
            print(*args,file=f)
        if also_print:print(*args)
    return write


def calculate_edge(all_samples_edges_relation,y,edge):
    y0_edge=[]
    y1_edge=[]
    for i in range(len(all_samples_edges_relation)):
        temp_dic=all_samples_edges_relation[i]
        value=0
        if edge in temp_dic:
           value=temp_dic[edge]

        if y[i]==0:
            y0_edge.append(value)
        else:
            y1_edge.append(value)


    return  y0_edge, y1_edge

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def cellTypeTrans(str,cellType):

    strs=str.split('-')
    T_str0=cellType[int(strs[0])-1]
    T_str1= cellType[int(strs[1])-1]
    return T_str0+'<--->'+T_str1

if __name__ =="__main__":
    log = saveToFile("./result.txt", also_print=True)
    top_edge=30;
    y0_distinguish_all =[];
    y1_distinguish_all =[];
    mat_folder='.\\matFile\\'
    labelPath='.\\label.xlsx'
    ls = [os.path.join(mat_folder, i) for i in sorted(os.listdir(mat_folder))]
    y_index = 1
    create_graph_method = "knn"
    k_nebs = 10
    sortEdgesR = True
    cellType = ['Fibroblasts', 'B7H4+Monocytes', 'BCells', 'Endothelials', 'Epithelials',
                'EpithelialsWithCytotoxicCells', 'Fibroblasts', 'NK', '1L-6ProducingCells',
                'ImmuneSuppresiveTCells', 'InfiltratingMacrophages',
                'Lineage', 'Monocytes', 'NK', 'Neutrophils', 'PD-1+ImmuneSuppresiveCells', 'ResidentMacrophages',
                'TCells', 'TNFα+Epithelials', 'Tregs']

    use_MaxMinNorm = True
    myList=[]
    y=[]

    all_acc_train=[]
    all_acc_test=[]
    all_acc_val=[]

    all_f1_train=[]
    all_f1_test=[]
    all_f1_val=[]

    all_AUC_train = []
    all_AUC_test = []
    all_AUC_val = []

    for i in range(1,6):
        train_list=sio.loadmat('split'+'_'+str(i)+'.mat')['train_idx']
        test_list =np.sort(sio.loadmat('split' + '_' + str(i) + '.mat')['test_idx'])
        val_list=sio.loadmat('split' + '_' + str(i) + '.mat')['val_idx']
        setup_seed(666)


        train_set=IMC_dataset(
              mat_folder=mat_folder,
              excel_path=labelPath,
              create_graph_method=create_graph_method,
              k_nebs=k_nebs,
              use_MaxMinNorm=use_MaxMinNorm, # 特征归一化
              mat_list=train_list)

        test_set = IMC_dataset(
             mat_folder=mat_folder,
             excel_path=labelPath,
             create_graph_method=create_graph_method,
             k_nebs=k_nebs,
             use_MaxMinNorm=use_MaxMinNorm,
             mat_list=test_list)

        val_set = IMC_dataset(
             mat_folder=mat_folder,
             excel_path=labelPath,
             create_graph_method=create_graph_method,
             k_nebs=k_nebs,
             use_MaxMinNorm=use_MaxMinNorm,
             mat_list=val_list)





        train_results_AUC,val_results_AUC,test_results_AUC,train_results_f1,val_results_f1,test_results_f1,train_results_acc,val_results_acc,test_results_acc,optimNet=train(
                net=GatNet(
                    input=10,
                    hiddens=[[16,1,0.8],[8,1,0.8]],
                    classifier=nn.Sequential(nn.Linear(8,1),nn.Sigmoid())
                ),
                dataset=train_set,
                testset=test_set,
                valset=val_set,
                y_index=y_index,
                num_epochs=200,
                learning_rate=1e-2,
                batch_size=len(train_set.tags),
                device=torch.device("cpu"),
                verbose=True,
                measure=True
        )

        all_acc_train.append(train_results_acc)
        all_acc_test.append(test_results_acc)
        all_acc_val.append(val_results_acc)

        all_f1_train.append(train_results_f1)
        all_f1_test.append(test_results_f1)
        all_f1_val.append(val_results_f1)

        all_AUC_train.append(train_results_AUC)
        all_AUC_test.append(test_results_AUC)
        all_AUC_val.append(val_results_AUC)
        AUC_train,Acc_train, _, all_samples_edges_relation, _, _, _,all_samples_edges_weight= test(
            model=optimNet,
            dataset=train_set,
            y_index=y_index,
            batch_size=len(train_set.tags),
            device=torch.device("cpu"),
            get_erelation=True
        )


    log("test  acc : avg : {}, std : {}".format(np.average(all_acc_test),np.std(all_acc_test)))
    #
    log("test  f1 : avg : {}, std : {}".format(np.average(all_f1_test),np.std(all_f1_test)))
    #
    log("test  AUC : avg : {}, std : {}".format(np.average(all_AUC_test),np.std(all_AUC_test)))




