from IMC_dataset import edges_relation_by_type_y, sort_dict


def edges_relation_by_type_y(edges_relation: list, ys: list, topK=None):
    ans = [{}, {}]

    for e, y in zip(edges_relation, ys):

        for i in dict(sort_dict(e, sort=True, topK=topK)):
            if i in ans[y]:
                ans[y][i] += e[i]
            else:
                ans[y][i] = e[i]
    return (ans[0], ans[1])

def sort_dict(d:dict,sort=None,topK=None):
    if sort is None:return d
    if type(topK)==type(0.1):topK=int(len(d.items())*topK)
    return sorted(d.items(),key=lambda x:x[1],reverse=sort)[:topK]


def cellTypeTrans(str,cellType):

    strs=str.split('-')
    T_str0=cellType[int(strs[0])-1]
    T_str1= cellType[int(strs[1])-1]
    return T_str0+'<--->'+T_str1

