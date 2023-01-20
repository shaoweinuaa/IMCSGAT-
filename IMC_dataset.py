#%% 
from typing import Any, Union
import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
import os
from tqdm import tqdm
import dgl
from torch.utils.data import DataLoader
from scipy.spatial import distance

def rp(path)->str:# relative path
    try:return os.path.join(os.path.dirname(__file__),path)
    except NameError:return path

def createGraphByKnn(pos,k):
    def knn(pos,k):

        def calc_dist(x0,y0,x1,y1):

            return (x0-x1)**2+(y0-y1)**2
        if pos.shape[0]<k:
            print("can not construct graph")
            raise Exception("can not construct graph")
        
        Dist=np.zeros((pos.shape[0],pos.shape[0]))

        for i in range(pos.shape[0]):
            j=i
            while j<pos.shape[0]:
                Dist[i][j]=Dist[j][i]=calc_dist(pos[i][0],pos[i][1],pos[j][0],pos[j][1])
                j+=1
        edges=[]
        for i in range(pos.shape[0]):
            maxKArgs=np.argpartition(Dist[i],k+1)[:k+1]
            for j in maxKArgs:
                if i==j:continue
                edges.append([i,j])
        return np.array(edges)

    def generateNet(edges):
        _edges=torch.from_numpy(edges[:,0]),torch.from_numpy(edges[:,1])
        g=dgl.graph(_edges,num_nodes=len(pos))
        return g

    return generateNet(knn(pos,k))

def createGraphByTopKEdges(pos: list, k: Union[int, float]) -> dgl.DGLGraph:
    len_pos = len(pos)
    if k < 1 or (k == 1 and type(k) != type(int)):
        k = int((len_pos**2)*k)
    dist = distance.cdist(pos, pos, 'euclidean')
    edges = np.array(list(
        filter(
            lambda x:x[0]!=x[1],
            map(
                lambda x: [x[0]//len_pos, x[0] % len_pos],
                sorted(list(enumerate(dist.flatten())), key=lambda a: a[1])
            )
        )
    ))[:k]
    e1, e2 = torch.from_numpy(edges[:, 0]), torch.from_numpy(edges[:, 1])
    g = dgl.graph((e1, e2), num_nodes=len(pos),idtype=torch.int64)
    return g

def createGraphByTopKNodes(pos:np.ndarray,k:Any,node_types):
    pos_with_ind = np.column_stack((list(range(len(pos))),pos))
    def handle_node_i(i):
        i=int(i)
        i_type=node_types[int(i)]
        edges=[]
        diff_type_pos = pos_with_ind[node_types!=i_type]
        dist = distance.cdist([pos[i]], diff_type_pos[:,1:], 'euclidean')
        dist = dist[0]
        select_indexs= np.argsort(dist)[:k]
        real_indexs = diff_type_pos[select_indexs][:,0]
        real_indexs = np.array(real_indexs,dtype=np.int32)
        edges.extend(list(zip([i]*len(real_indexs),real_indexs)))
        return np.array(edges,dtype="object")
    v_handle_node_i = np.vectorize(handle_node_i)
    edges = v_handle_node_i(pos_with_ind[:,0])
    edges = np.concatenate(edges)
    edges = np.array(edges,dtype=np.int32)
    _edges=torch.from_numpy(edges[:,0]),torch.from_numpy(edges[:,1])
    g=dgl.graph(_edges,num_nodes=len(pos),idtype=torch.int64)
    return g

def __createGraphByTopKNodes(pos:np.ndarray,k:Any,node_types:np.ndarray):


    pos_with_ind = np.column_stack((list(range(len(pos))),pos))
    node_types_set=set(node_types)
    def handle_node_i(i):
        i_type=node_types[i]
        diff_types = node_types_set - {i_type}

        edges=[]
        for diff_type in diff_types:
            diff_type_pos = pos_with_ind[node_types==diff_type] #
            dist = distance.cdist([pos[i]], diff_type_pos[:,1:], 'euclidean') #
            dist = dist[0]
            select_indexs= np.argsort(dist)[:k]
            real_indexs = diff_type_pos[select_indexs][:,0]
            edges.extend(list(zip([i]*len(real_indexs),real_indexs)))
        return np.array(edges,dtype="object")
    v_handle_node_i = np.vectorize(handle_node_i)
    edges = v_handle_node_i(pos_with_ind[:,0])
    edges = np.concatenate(edges)
    edges = np.array(edges,dtype=np.int32)
    _edges=torch.from_numpy(edges[:,0]),torch.from_numpy(edges[:,1])
    g=dgl.graph(_edges,num_nodes=len(pos))
    return g


class IMC_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mat_folder=None,
        mat_list=None,
        excel_path=None,
        k_nebs=None,
        create_graph_method=None,
        use_MaxMinNorm=True,

    )->None:

        super().__init__()
        self.tags=[]
        self.gs=[]
        self.y1s=[]

        self.mat_name=[]

        def get_y_func(excel_path)->object:
            xls=pd.read_excel(excel_path,header=None,names=["tag","y1"])
            def get_y(tag)->list:

                ans=xls[xls["tag"]==tag].to_numpy()

                if len(ans)!=1:raise Exception()

                return ans[0]
            return get_y
        def handle_each_mat(mat_path)->tuple:
            tag=os.path.basename(str(mat_path))[:-6]
            mat=loadmat(mat_path[0])["Node"]
            node_types=np.int32(mat[:,0])
            poss=mat[:,1:3]
            A = [3, 4, 5, 6, 7, 8, 15, 16, 17, 21]
            feats = mat[:, A]

            try:
                if create_graph_method == "knn":g=createGraphByKnn(pos=poss,k=k_nebs)
                elif create_graph_method == "node":g=createGraphByTopKNodes(pos=poss,k=k_nebs,node_types=node_types)
                elif create_graph_method == "edge":g=createGraphByTopKEdges(pos=poss,k=k_nebs)
                else:raise Exception('Can not construct the graph')
            except:
                raise Exception("Wrong Path ")
            g.ndata['node_type']=torch.from_numpy(node_types).to(torch.int32)
            def MaxMinNorm(array):
                maxcols=array.max(axis=0)
                mincols=array.min(axis=0)
                data_shape = array.shape
                data_rows = data_shape[0]
                data_cols = data_shape[1]
                t=np.empty((data_rows,data_cols))
                for i in range(data_cols):
                    t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
                return t
            if use_MaxMinNorm:feats=MaxMinNorm(feats)
            g.ndata['feats']=torch.from_numpy(feats).to(torch.float32)
            return tag,g
        ls=mat_list if mat_list is not None else [os.path.join(mat_folder,i) for i in sorted(os.listdir(mat_folder))]
        ls=ls[0]
        get_y=get_y_func(excel_path)
        for e in tqdm(ls):
            tag,g=handle_each_mat(e)
            _,y1=get_y(tag)
            self.tags.append(tag)
            self.gs.append(g)
            self.y1s.append(y1)
            self.mat_name.append(os.path.basename(str(e))[:-6])

    def __len__(self)->int:return len(self.tags)
    def __getitem__(self,index)->tuple:
        return  self.tags[index],\
                self.gs[index],\
                self.y1s[index],\
                self.mat_name[index]

def collate_fn_imc(batchs):
    ans=[ [] for i in range(len(batchs[0]))]
    for batch in batchs:
        for i in range(len(batch)):
            ans[i].append(batch[i])
    return ans


def sort_dict(d:dict,sort=None,topK=None):
    if sort is None:return d
    if type(topK)==type(0.1):topK=int(len(d.items())*topK)
    return sorted(d.items(),key=lambda x:x[1],reverse=sort)[:topK]



def get_edges_relation(g:dgl.DGLGraph,atten):
    a,b=g.edges()
    relation_dct={}
    weight_dict_sum={}
    weight_dict={}
    for i in range(len(a)):
        r1,r2 = g.ndata['node_type'][a[i]], g.ndata['node_type'][b[i]]
        if r2<r1 : r1,r2=r2,r1
        relation_str = "{}-{}".format(r1,r2)
        if relation_str in relation_dct:
            relation_dct[relation_str]+=1
            weight_dict_sum[relation_str]+=atten[i]
        else:
            relation_dct[relation_str]=1
            weight_dict_sum[relation_str]=atten[i]

    for str in relation_dct.keys():
        weight_dict[str]=weight_dict_sum[str][0][0].numpy()/relation_dct[str]


    return relation_dct, weight_dict

def edges_relation_by_type_y(edges_relation: list, ys: list, topK=None):
    ans = [{}, {}]

    for e, y in zip(edges_relation, ys):

        for i in dict(sort_dict(e, sort=True, topK=topK)):

            if i in ans[y]:

                ans[y][i] += e[i]
            else:

                ans[y][i] = e[i]

    return (ans[0], ans[1])

