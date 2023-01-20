import numpy as np
import torch
from tqdm import tqdm
from IMC_dataset import IMC_dataset,DataLoader,collate_fn_imc,get_edges_relation
from gat import GatNet
import torch.nn as nn
import sys
import dgl
from sklearn.metrics import f1_score
from sklearn import metrics




def test(model,dataset,y_index:int,batch_size,device,get_erelation=False):
    model.eval()
    imc_dataloader=DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_imc
    )
    loss = nn.MSELoss()
    epoch_loss=0
    correct=0
    total=0
    all_samples_edges_relation=[]
    all_samples_edges_weight=[]
    all_mats_names=[]
    all_ys=[]
    with torch.no_grad():
        for tags,gs,y1s,mats_names in imc_dataloader:
            if y_index==1:
                y=torch.tensor(y1s,dtype=torch.float32).to(device)

            else:raise Exception("y_index <- {1,2,3},but got {}".format(y_index))

            if get_erelation==False:
                g=dgl.batch(gs).to(device)
                _g,_code,y_pred,_=model(g)
            else:
                y_pred=[]
                for i in range(len(tags)):
                    g=gs[i].to(device)
                    _g,_code,_y_pred,atten=model(g)
                    y_pred.append(_y_pred)
                    each_sample_edges_relation, each_sample_edge_weight = get_edges_relation(_g,atten)
                    all_samples_edges_relation.append(each_sample_edges_relation)
                    all_samples_edges_weight.append(each_sample_edge_weight)
                y_pred=torch.tensor(y_pred).to(device)
                all_mats_names.extend(mats_names)
                all_ys.extend([int(i) for i in y.detach().cpu().tolist()])

            y_pred=y_pred.reshape(-1)
            epoch_loss+=loss(y_pred,y).item()
            y_pred_label=[0 if i<0.5 else 1 for i in y_pred.cpu().detach().numpy()]
            y_label=np.array(y.detach().cpu().numpy(),dtype=int)
            y_pred[np.isnan(y_pred.cpu().detach().numpy())] = 0
            fpr, tpr, thresholds = metrics.roc_curve(y_label, y_pred, pos_label=1)
            AUC=metrics.auc(fpr, tpr)

            correct+=np.sum(y_pred_label==y_label)
            total+=len(y_pred_label)
            f1=f1_score(y_label,y_pred_label)
    if get_erelation:return AUC,correct/total, f1, all_samples_edges_relation, all_mats_names, all_ys,y_label,all_samples_edges_weight
    return AUC,correct/total, f1,all_samples_edges_relation, all_mats_names, all_ys, y_pred_label



def train(
    net:GatNet,
    num_epochs:int,
    learning_rate:float,
    batch_size:int,
    dataset:IMC_dataset,
    testset:IMC_dataset,
    valset:IMC_dataset,
    y_index:int,
    device:str,
    verbose=True,
    measure=False
):
    Net = [];
    net=net.to(device)
    optimizer=torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=0
    )
    loss = nn.MSELoss()
    train_f1=np.zeros([num_epochs])
    val_f1=np.zeros([num_epochs])
    test_f1=np.zeros([num_epochs])


    train_acc=np.zeros([num_epochs])
    val_acc=np.zeros([num_epochs])
    test_acc=np.zeros([num_epochs])

    train_AUC = np.zeros([num_epochs])
    val_AUC = np.zeros([num_epochs])
    test_AUC = np.zeros([num_epochs])


    test_label_all=np.zeros([num_epochs,len(testset.tags)])
    train_label_all = np.zeros([num_epochs, len(dataset.tags)])
    val_label_all=np.zeros([num_epochs, len(valset.tags)])

    with tqdm(total=num_epochs,file=sys.stdout) as pbar:
        for epoch in range(num_epochs):
            net.train()
            imc_dataloader=DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn_imc
            )
            epoch_loss=0
            for tags,gs,y1s,mats_names in imc_dataloader:
                optimizer.zero_grad()
                if y_index==1:
                    y=torch.tensor(y1s,dtype=torch.float32).to(device)

                else:raise Exception("y_index <- {1,2,3},but got {}".format(y_index))

                g=dgl.batch(gs).to(device)
                _g,_code,y_pred,_=net(g)


                y_pred=y_pred.reshape(-1)
                l=loss(y_pred,y)
                print("epoch:"+str(epoch)+' loss:'+str(l.detach()))
                epoch_loss+=l.item()
                l.backward()
                optimizer.step()
            if verbose:
                if epoch%1==0 or epoch==num_epochs-1 :
                    Net.append(net)
                    if measure!=False:
                        train_AUC[epoch],train_acc[epoch],train_f1[epoch],_,_,_,train_label_all[epoch,:] = test(
                            model=net,
                            dataset=dataset,
                            y_index=y_index,
                            batch_size=len(dataset.tags),
                            device=torch.device("cpu")
                        )


                        test_AUC[epoch],test_acc[epoch], test_f1[epoch],_, _, _,test_label_all[epoch,:]= test(
                            model=net,
                            dataset=testset,
                            y_index=y_index,
                            batch_size=len(testset.tags),
                            device=torch.device("cpu")
                        )


                        val_AUC[epoch],val_acc[epoch], val_f1[epoch],_, _, _ ,val_label_all[epoch,:]= test(
                            model=net,
                            dataset=valset,
                            y_index=y_index,
                            batch_size=len(valset.tags),
                            device=torch.device("cpu")
                        )
                        print("train_AUC:{%.3f}\t val_AUC:{%.3f} \t test_AUC:{%.3f}  \t train_acc:{%.3f}\t val_acc:{%.3f} \t test_acc:{%.3f}  \t train_f1:{%.3f} \t val_f1:{%.3f} \t test_f1:{%.3f}"%(train_AUC[epoch], val_AUC[epoch],test_AUC[epoch],train_acc[epoch], val_acc[epoch],test_acc[epoch],train_f1[epoch], val_f1[epoch],test_f1[epoch]))


        pbar.update(1)
        val_index=np.argmax(val_AUC)

        optim_Net = Net[val_index]

        train_results_AUC=train_AUC[val_index]
        test_results_AUC=test_AUC[val_index]
        val_results_AUC=val_AUC[val_index]


        train_results_f1=train_f1[val_index]
        test_results_f1=test_f1[val_index]
        val_results_f1=val_f1[val_index]

        train_results_acc=train_acc[val_index]
        test_results_acc=test_acc[val_index]
        val_results_acc=val_acc[val_index]


        test_pre_label=test_label_all[val_index,:]



        print(test_results_AUC,test_results_acc,test_results_f1,test_pre_label)
    return train_results_AUC,val_results_AUC,test_results_AUC,train_results_f1,val_results_f1,test_results_f1,train_results_acc,val_results_acc,test_results_acc,optim_Net

