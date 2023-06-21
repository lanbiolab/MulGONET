from keras.callbacks import LearningRateScheduler
from keras import backend as K
import random
import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold

from evaluates import evaluates,get_class_weight

from preprocessing import Get_Node_relationships,Get_pathway_gene_relationships,Preprocessing,gene_pathways_matrix

from MulGONET import create_model

# 定义一个学习率更新函数
def myScheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 70 == 0 and epoch != 0:
        # 获取在model.compile()中设置的学习率lr
        lr = K.get_value(models.optimizer.lr)
        # 按照lr * 0.1，重新更新学习率
        K.set_value(models.optimizer.lr, lr * 0.5)
    return K.get_value(models.optimizer.lr)


# 定义一个学习率的回调函数
myReduce_lr = LearningRateScheduler(myScheduler)





#获取pathway-pathway relations
Get_Node_relation,pathway_union_bp  = Get_Node_relationships('biological_process','./data/GO_hierarchical_structure.csv','bp')

Get_Node_relation_mf,pathway_union_mf = Get_Node_relationships('molecular_function','./data/GO_hierarchical_structure.csv','mf')

gene_data_bp = Get_pathway_gene_relationships('bp')

gene_data_mf = Get_pathway_gene_relationships('mf')



#获取数据集
meth_data, cnv_amp, cnv_del, exp_data, response = Preprocessing(Get_Node_relation,Get_Node_relation_mf,gene_data_bp,gene_data_mf)

#类权重系数
x_0 , x_1 = get_class_weight(response)

gene_pathway_bp_dfss,gene_pathway_mf_dfss = gene_pathways_matrix(meth_data, cnv_amp, cnv_del, exp_data,pathway_union_bp,pathway_union_mf,gene_data_bp,gene_data_mf)



multi_data = pd.concat([meth_data,cnv_amp,cnv_del,exp_data],axis = 1)
response = response.loc[multi_data.index]



skf = StratifiedKFold(n_splits=5,shuffle=True) #,shuffle=True class_weight = {0:x_0,1:x_1},random_state=1029)

multi_data_train = multi_data.values
multi_data_test = response['response'].values



total_score = []

for i in range(0,1):
    kfscore = []
    p= 0
    for train_index, test_index in skf.split(multi_data_train,multi_data_test):

        multi_data_train_x, multi_data_test_x = multi_data_train[train_index], multi_data_train[test_index] #突变数据的划分  x 自变量


        multi_data_train_y, multi_data_test_y = multi_data_test[train_index], multi_data_test[test_index]   #  y 应变量


         #,class_weight = {0:x_0,1:x_1},
        opt = keras.optimizers.Adam(lr = 0.001)

        models = create_model(multi_data_train_x,gene_pathway_bp_dfss,Get_Node_relation, gene_pathway_mf_dfss,Get_Node_relation_mf,
                 bp_net=False, mf_net=False, optimizers=opt)

        #     keras.utils.plot_model(models, show_shapes=True)   callbacks=[myReduce_lr],

        models.fit(multi_data_train_x,multi_data_train_y,
                          epochs=140,batch_size = 64,class_weight = {0:x_0,1:x_1},callbacks=[myReduce_lr],
                          validation_data=( multi_data_test_x,multi_data_test_y)
                  )

        y_pred = models.predict(multi_data_test_x)


        print(evaluates(multi_data_test_y, y_pred))

        kfscore.append(evaluates(multi_data_test_y, y_pred))

#         get_important_score(p,models)

#         p = p+1
    #平均值
    kfscore = np.array(kfscore).sum(axis= 0)/5.0     #pre,acc,rec,auc
    print(kfscore)
    total_score.append(kfscore)
total_score



