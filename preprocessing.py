import pandas as pd
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from sklearn.utils import shuffle
#特征选择
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
random.seed(10290000)

# 通路数据加载from sklearn.utils import shuffle
# GO 通路数据加载   #BP通路


# get nodes at layers
def get_nodes_at_level(net, distance,roots):
    nodes = set(nx.ego_graph(net, roots[0], radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, roots[0], radius=distance - 1))
    return list(nodes)


# acquire the whole node of network,  include the all  layers
def get_nodes(net, num,roots):
    net_nodes = []  #   store  any nodes

    for i in range(1, num + 1):
        net_nodes.append(get_nodes_at_level(net, i,roots))

    return net_nodes



#  #读取单个节点跟下一层的连接
def add_nodes(net, net_nodes):
    for i in range(len(net_nodes) - 2, -1, -1):

        data_temp = copy.deepcopy(net_nodes[i])

        for n in net_nodes[i]:
            nexts = net.successors(n)  # 对n的后继搜索
            temp = [nex for nex in nexts]
            if len(temp) == 0:
                data_temp.remove(n)
            elif len(set(temp).intersection(set(net_nodes[i + 1]))) == 0:
                data_temp.remove(n)
            else:
                continue
        net_nodes[i] = data_temp
    return net_nodes



# create relationship
def get_note_relation(net,net_nodes):
    node_mat = []
    # 读取单个节点跟下一层的连接
    for i in range(len(net_nodes) - 1):
        dicts = {}
        for n in net_nodes[i]:
            nexts = net.successors(n)  # 对n的后继搜索
            x = [nex for nex in nexts if nex in net_nodes[i + 1]]
            dicts[n] = x

        mat = np.zeros((len(net_nodes[i]), len(net_nodes[i + 1])))  # 生成矩阵 【行 列】
        for p, gs in dicts.items():  ## 得到key 和 values
            g_inds = [net_nodes[i + 1].index(g) for g in gs]
            p_ind = net_nodes[i].index(p)
            mat[p_ind, g_inds] = 1

        df = pd.DataFrame(mat, index=net_nodes[i], columns=net_nodes[i + 1])
        node_mat.append(df.T)
    return node_mat




def Get_pathway_gene_relationships(gene_type):


    #  genes_pathways_relationships in the bp_network
    if gene_type == 'bp':

        bp_url = './data/gene_data_bp.npy'

        gene_data_bp = np.load(bp_url).item()

        for keys in list(gene_data_bp):
             if len(gene_data_bp[keys]) > 200:
                gene_data_bp.pop(keys)

        return gene_data_bp

    elif gene_type == 'mf':

        # genes_pathways_relationships in the mf_network
        mf_url = './data/gene_data_mf.npy'

        gene_data_mf = np.load(mf_url).item()

        for keys in list(gene_data_mf):
            if len(gene_data_mf[keys]) > 200:
                gene_data_mf.pop(keys)


        return  gene_data_mf



#return  modeled the relationships
def Get_Node_relationships(data_type,data_url,gene_type):

    data = pd.read_csv(data_url)

    data = data[data['2'] == data_type]

    data = data[['0', '1']]

    data.columns = ['parent', 'child']
    human_hierarchy = data[data['child'].str.contains('GO')]  # 选出HSA 的通路

    net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent',
                                  create_using=nx.DiGraph())  ##创建空的简单有向图   pandas的datafrmae每一行代表一个连接里面的点
    net.name = 'GO'

    roots = [n for n, d in net.in_degree() if d == 0]

    print(roots)

    net_num = 5  #define 5 layer pathway network

    net_nodes = get_nodes(net, net_num,roots)

    net_nodes = add_nodes(net, net_nodes)


    Get_Node_relation = get_note_relation(net,net_nodes)

    gene_data_type = list(Get_pathway_gene_relationships(gene_type))

    pathway_union = list(set(Get_Node_relation[3].index).intersection(set(gene_data_type)))

    print(len(pathway_union))

    Get_Node_relation[3] = Get_Node_relation[3].loc[pathway_union]

    print(Get_Node_relation[0].shape)
    print(Get_Node_relation[1].shape)
    print(Get_Node_relation[2].shape)
    print(Get_Node_relation[3].shape)

    return Get_Node_relation, pathway_union




def get_raw_data():


    print('loading data')

    # meth
    meth_data = pd.read_csv("E:\\Article_data\\TCGA_BLCA\\TCGA-BLCA_meth.csv", index_col=0)

    # cnv_data
    cnv_data = pd.read_csv("E:\\Article_data\\TCGA_BLCA\\TCGA-BLCA_cnv.csv", index_col=0)

    # exp_data
    exp_data = pd.read_csv("E:\\Article_data\\TCGA_BLCA\\TCGA_BLCA_fpkm_expression.csv", index_col=0)

    # 复发的status
    response = response = pd.read_csv('E:\\Article_data\\TCGA_BLCA\\new_tumor_event_response.csv',
                                      index_col=0)
    response = response[['response']]

    # 打乱数据集
    meth_data = shuffle(meth_data)
    cnv_data = shuffle(cnv_data)
    response = shuffle(response)
    exp_data = shuffle(exp_data)

    return meth_data, cnv_data, exp_data,response



def fesature_select(data,response,select_k):



    print('starting feature_selection')
    # 基因表达
    temp_data = data.join(response, how='inner')

    # 选择K个最好的特征，返回选择特征后的数据
    model = SelectKBest(chi2, k=select_k)

    train_select = model.fit_transform(temp_data.values[:, 0:-1], temp_data.values[:, -1])

    features = model.get_support()

    data = data.loc[:, features]

    return data







def Preprocessing(Get_Node_relation,Get_Node_relation_mf,gene_data_bp,gene_data_mf):


    meth_data, cnv_data, exp_data, response = get_raw_data()


    # 基因
    protein_gene = pd.read_csv(
        'E:\\Article_data\\Gene Ontology\\protein-coding_gene_with_coordinate_minimal.txt', sep='\t',
        header=None)


    protein_exp_genes = list(set(protein_gene[3]).intersection(set(exp_data.columns)))
    exp_data = exp_data[protein_exp_genes]
    # 甲基化
    protein_meth_genes = list(set(protein_gene[3]).intersection(set(meth_data.columns)))
    meth_data = meth_data[protein_meth_genes]


    genes_existed_pathway_bg = []
    genes_existed_pathway_mf = []


    for index in Get_Node_relation[3].index:
        genes_existed_pathway_bg = genes_existed_pathway_bg + list(gene_data_bp[index])
    genes_existed_pathway_bg = set(genes_existed_pathway_bg)

    for index in Get_Node_relation_mf[3].index:
        genes_existed_pathway_mf = genes_existed_pathway_mf + list(gene_data_mf[index])
    genes_existed_pathway_mf = set(genes_existed_pathway_mf)

    finall_gene = list(genes_existed_pathway_bg) + list(genes_existed_pathway_mf)

    meth_data = meth_data[list(set(meth_data.columns).intersection(finall_gene))]
    cnv_data = cnv_data[list(set(cnv_data.columns).intersection(finall_gene))]
    exp_data = exp_data[list(set(exp_data.columns).intersection(finall_gene))]


    # 过滤表达接近0的数据
    t = exp_data.describe()
    exp_data = exp_data[t.loc['75%'][t.loc['75%'] > 0.05].index]

    # 拷贝数数据拷贝
    cnv_amp = copy.deepcopy(cnv_data)  # cnv zmp
    cnv_del = cnv_data  # cnv del

    # 拷贝数删除
    cnv_del[cnv_del >= 0.0] = 0.0
    cnv_del[cnv_del < 0.0] = 1.0

    # 拷贝数增加
    cnv_amp[cnv_amp <= 0.0] = 0.0
    cnv_amp[cnv_amp > 0.0] = 1.0

    # set(meth_data.index),
    # 选取共同的样本
    sample = set.intersection(set(meth_data.index), set(cnv_amp.index), set(cnv_del.index), set(response.index),
                              set(exp_data.index))  # 三样本相同的sample

    # sample = set.intersection(set(meth_data.index), set(cnv_amp.index),set(cnv_del.index),set(response.index))
    print('sample_num = ', len(sample))

    meth_data = meth_data.loc[sample]

    cnv_amp = cnv_amp.loc[sample]

    cnv_del = cnv_del.loc[sample]

    exp_data = exp_data.loc[sample]

    response = response.loc[sample]





    # feature selecting
    meth_data = fesature_select(meth_data, response, 1000)
    cnv_amp = fesature_select(cnv_amp, response, 1000)
    cnv_del = fesature_select(cnv_del, response, 1000)
    exp_data = fesature_select(exp_data, response, 1000)


    #对基因表达做归一化
    #数据归一化

    scaler = MinMaxScaler() #实例化
    scaler = scaler.fit(exp_data) #fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(exp_data) #通过接口导出结果
    exp_data = pd.DataFrame(result,index =exp_data.index,columns = exp_data.columns )

    print('meth_data shape = ',meth_data.shape)
    print('cnv_amp shape = ',cnv_amp.shape)
    print('cnv_del shape = ',cnv_del.shape)
    print('exp_data shape = ',exp_data.shape)
    print('response shape = ',response.shape)

    return meth_data, cnv_amp, cnv_del, exp_data, response






def gene_pathways_matrix(meth_data, cnv_amp, cnv_del, exp_data,pathway_union_bp,pathway_union_mf,gene_data_bp,gene_data_mf):


    union_gene_meth = list(meth_data.columns)
    union_gene_amp = list(cnv_amp.columns)
    union_gene_del = list(cnv_del.columns)
    union_gene_exp = list(exp_data.columns)



    mask_list = [union_gene_meth, union_gene_amp, union_gene_del, union_gene_exp]

    gene_pathway_bp_dfs = []
    gene_pathway_mf_dfs = []

    # 第一个网络
    for i in range(len(mask_list)):
        pathways_genes = np.zeros((len(pathway_union_bp), len(mask_list[i])))  # 生成矩阵 【行 列】
        for p in pathway_union_bp:
            gs = gene_data_bp[p]
            g_inds = [mask_list[i].index(g) for g in gs if g in mask_list[i]]
            p_ind = pathway_union_bp.index(p)
            pathways_genes[p_ind, g_inds] = 1
        gene_pathway_bp = pd.DataFrame(pathways_genes, index=pathway_union_bp, columns=mask_list[i])

        # 丢掉不在通路上的基因
        #     gene_pathway_bp = gene_pathway_bp.loc[:, (gene_pathway_bp != 0).any(axis=0)]
        gene_pathway_bp_dfs.append(gene_pathway_bp)

    # 第二个网络
    for i in range(len(mask_list)):
        pathways_genes = np.zeros((len(pathway_union_mf), len(mask_list[i])))  # 生成矩阵 【行 列】
        for p in pathway_union_mf:
            gs = gene_data_mf[p]
            g_inds = [mask_list[i].index(g) for g in gs if g in mask_list[i]]
            p_ind = pathway_union_mf.index(p)
            pathways_genes[p_ind, g_inds] = 1
        gene_pathway_mf = pd.DataFrame(pathways_genes, index=pathway_union_mf, columns=mask_list[i])

        # 丢掉不在通路上的基因
        #     gene_pathway_mf = gene_pathway_mf.loc[:, (gene_pathway_mf != 0).any(axis=0)]
        gene_pathway_mf_dfs.append(gene_pathway_mf)



        # four data integration

    gene_pathway_bp_dfss = pd.concat([gene_pathway_bp_dfs[0],gene_pathway_bp_dfs[1],gene_pathway_bp_dfs[2],gene_pathway_bp_dfs[3]],axis = 1)
    gene_pathway_mf_dfss = pd.concat([gene_pathway_mf_dfs[0],gene_pathway_mf_dfs[1],gene_pathway_mf_dfs[2],gene_pathway_mf_dfs[3]],axis = 1)



    print('gene_pathway_bp_dfss shape=',gene_pathway_bp_dfss.shape,'gene_pathway_mf_dfss shape=',gene_pathway_mf_dfss.shape)



    return gene_pathway_bp_dfss,gene_pathway_mf_dfss




