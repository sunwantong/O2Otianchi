from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
import numpy as np
import pandas as pd
import os as os
import sklearn.ensemble as sk
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import operator
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr



def loadDataSet(train,validate,test):
    change_path_utils(train)
    train_set = pd.read_csv(os.path.basename(train),sep=",")

    change_path_utils(validate)
    validate_set = pd.read_csv(os.path.basename(validate), sep=",")

    change_path_utils(test)
    test_set = pd.read_csv(os.path.basename(test), sep=",")
    return train_set,validate_set,test_set


def evaluate_method(train_set,validate_set,test_set):

    ################xgboost#######################
    pred_value = xgb_model(train_set, validate_set, test_set)
    tess = test_set[["user_id","coupon_id","date_received"]]
    a = pd.DataFrame(pred_value, columns=["prob"])
    res = pd.concat([tess, a["prob"]], axis=1)
    res.to_csv("d://aa.csv", index=None)


    ##################RandomForest################
    # pred_value = random_forest_model(train_set, validate_set, test_set)
    # tess = test_set[["user_id","coupon_id","date_received"]]
    # a = pd.DataFrame(pred_value, columns=["prob_pos","prob_neg"])
    # res = pd.concat([tess, a["prob_neg"]], axis=1)
    # res.to_csv("f://resultrf.csv", index=None)

    ##################GBDT###########################
    # pred_value = gradient_boost_regression_tree(train_set, validate_set, test_set)
    # tess = test_set[["user_id", "coupon_id", "date_received"]]
    # a = pd.DataFrame(pred_value, columns=["prob_pos", "prob_neg"])
    # res = pd.concat([tess, a[["prob_pos","prob_neg"]]], axis=1)
    # res.to_csv("f://resultgbtree.csv", index=None)



def gradient_boost_regression_tree(train_set, validate_set, test_set):
    train_y = train_set.label
    train_x = train_set.drop(
        ['label', "user_id", "coupon_id", "date_received", "merchant_id", "distance", "discount_rate"],
        axis=1)

    test_set_no_key = test_set.drop(
        ["user_id", "coupon_id", "date_received", "merchant_id", "distance", "discount_rate"],
        axis=1)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=7, random_state=0)
    clf.fit(train_x.values, train_y.values)
    pred_value = clf.predict_proba(test_set_no_key.values)
    return pred_value



def random_forest_model(train_set, validate_set, test_set):
    train_y = train_set.label
    train_x = train_set.drop(
        ['label', "user_id", "coupon_id", "date_received", "merchant_id", "distance", "discount_rate"],
        axis=1)

    val_y = validate_set.label
    val_X = validate_set.drop(
        ['label', "user_id", "coupon_id", "date_received", "merchant_id", "distance", "discount_rate"],
        axis=1)

    test_set_no_key = test_set.drop(
        ["user_id", "coupon_id", "date_received", "merchant_id", "distance", "discount_rate"],
        axis=1)

    rf = sk.RandomForestClassifier(max_depth=15)
    rf.fit(train_x.values, train_y.values)
    pred_value = rf.predict_proba(test_set_no_key.values)
    return pred_value

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def xgb_model(train_set, validate_set, test_set):
    train_y = train_set.label
    train_x = train_set.drop(['label',"user_id","coupon_id","date_received","merchant_id","distance","discount_rate"],
                             axis=1)

    #相关性分析
    # pearson_analysis_feature(train_x,train_y)

    val_y = validate_set.label
    val_X = validate_set.drop(['label',"user_id","coupon_id","date_received","merchant_id","distance","discount_rate"],
                              axis=1)


    xgb_val = xgb.DMatrix(val_X, label=val_y)
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    test_set_no_key = test_set.drop(["user_id","coupon_id","date_received","merchant_id","distance","discount_rate"],
                                    axis=1)
    xgb_test = xgb.DMatrix(test_set_no_key)
    ceate_feature_map(train_x)
    params = {  'booster':'gbtree',
                'objective': 'binary:logistic',  #多分类的问题
                # 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                'max_depth':5,  # 构建树的深度，越大越容易过拟合
                # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'subsample':0.7,  # 随机采样训练样本
                'colsample_bytree':0.7,  # 生成树时进行的列采样
                'min_child_weight':3,
                # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
                'silent':0,  #设置成1则没有运行信息输出，最好是设置为0.
                'eta': 0.05,  # 如同学习率
                'seed':1000,
                'nthread':7,  # cpu 线程数
                'eval_metric': 'auc' # 评价方式
              }

    plst = list(params.items())
    num_rounds = 1000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # early_stopping_rounds    当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds,watchlist)
    pred_value = model.predict(xgb_test)

    ###########重要性开始###########
    # importance = model.get_fscore(fmap='xgb.fmap')
    # importance = sorted(importance.items(), key=operator.itemgetter(1))
    # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # df['fscore'] = df['fscore'] / df['fscore'].sum()
    # plt.figure()
    # df.plot()
    # df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
    # plt.title('XGBoost Feature Importance')
    # plt.xlabel('relative importance')
    # # plt.gcf().savefig('feature_importance_xgb.png')
    # plt.show()
    ##########重要性结束############

    return pred_value



def process_data(train_set, validate_set, test_set):
    train_set.fillna(-1, inplace=True)
    validate_set.fillna(-1, inplace=True)
    test_set.fillna(-1, inplace=True)

    train_set.replace("null",-1, inplace=True)
    validate_set.replace("null",-1, inplace=True)
    test_set.replace("null",-1, inplace=True)

    # train_set.discount_man = train_set.discount_man.astype("float64")
    # train_set.discount_jian = train_set.discount_jian.astype("float64")
    #
    # validate_set.discount_man = validate_set.discount_man.astype("float64")
    # validate_set.discount_jian = validate_set.discount_jian.astype("float64")
    #
    # test_set.discount_man = test_set.discount_man.astype("float64")
    # test_set.discount_jian = test_set.discount_jian.astype("float64")

    #dfdfdfdfdf
    train_set.discount_jian_label_range = train_set.discount_jian_label_range.astype("float64")
    train_set.discount_man_label_range = train_set.discount_man_label_range.astype("float64")

    validate_set.discount_jian_label_range = validate_set.discount_jian_label_range.astype("float64")
    validate_set.discount_man_label_range = validate_set.discount_man_label_range.astype("float64")

    test_set.discount_jian_label_range = test_set.discount_jian_label_range.astype("float64")
    test_set.discount_man_label_range = test_set.discount_man_label_range.astype("float64")


    train_set.drop(["date_received_and_dates"],inplace=True,axis=1)
    validate_set.drop(["date_received_and_dates"],inplace=True,axis=1)
    test_set.drop(["date_received_and_dates"],inplace=True,axis=1)


    train_set.drop(["dates"], inplace=True, axis=1)
    validate_set.drop(["dates"], inplace=True, axis=1)
    test_set.drop(["dates"], inplace=True, axis=1)

    return train_set,validate_set,test_set


"""
1.分析每一个特征与响应变量的相关性
2.选择K个最好的特征，返回选择特征后的数据
3.第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，
    数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
4.参数k为选择的特征个数

"""

def pearson_analysis_feature(feature,label):
    # X = np.mat(X)
    # y = np.mat(y)
    label.to_csv("c://aa.csv",index=None)
    # res = SelectKBest(lambda X, Y: map(lambda x: pearsonr(x, label),feature), k=50).\
    #     fit_transform(feature,label)

    # res = SelectKBest(lambda X, Y: np.array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(feature,label)
    # res = map(lambda x: pearsonr(x,label),feature['user_get_total_count'])
    print(len(feature.columns))
    res = []
    for pos in range(len(feature.columns)):
        a = feature.iloc[:,pos]
        b = feature.iloc[:, pos].name
        print(b)
        x,y = pearsonr(a,label)
        print(x)
        res.append(x)
    # res = pearsonr(feature['user_get_total_count'],label)
    # a = sorted(res, reverse=True)
    # print(len(a))


if __name__ == "__main__":

    train_set, validate_set, test_set = loadDataSet(final_train_set, final_validate_set, final_test_set)

    train_set, validate_set, test_set = process_data(train_set, validate_set, test_set)

    evaluate_method(train_set, validate_set, test_set)


    # print(train_set.dtypes)




















#
# def offline_validate(train_set_x, train_set_y, validate_set):
#     validate_set_y_real = validate_set["label"].values
#     validate_set.drop(["label"], inplace=True, axis=1)
#
#     # pred_value = validate_mode(validate_set, train_set_x, train_set_y)
#     # validate_data = validate_set.values
#     pred_value = xgbs(train_set_x,train_set_y,validate_set.values)
#
#     fpr, tpr, thresholds = metrics.roc_curve(np.array(validate_set_y_real), np.array(pred_value))
#     auc = metrics.auc(fpr, tpr)
#
#     print("auc:",auc)
#
# def validate_mode(validate_set, train_set_x, train_set_y):
#     rf = sk.RandomForestClassifier(max_depth=15)
#     rf.fit(train_set_x, train_set_y)
#     pred_value = rf.predict(validate_set.values)
#     return pred_value
#
#
# def online_test(train_set_x, train_set_y,test_set):
#     pred_value = test_mode(test_set, train_set_x, train_set_y)
#     # pred_value = xgbs(train_set_x, train_set_y,test_set.values)
#
#     test_set = test_set[["user_id", "coupon_id", "date_received"]]
#
#     a = pd.DataFrame(pred_value, columns=["prob", "neg"])
#     res = pd.concat([test_set, a["prob"]], axis=1)
#     res.to_csv("d://result.csv", index=None)
#
#
# def test_mode(test_set, train_set_x, train_set_y):
#     rf = sk.RandomForestClassifier(max_depth=15)
#     rf.fit(train_set_x, train_set_y)
#     pred_value = rf.predict_proba(test_set.values)
#     return pred_value
#
#
# """
#  xgb模型
#
# """
# def xgbs(train_set_x,train_set_y,validate_set):
#     print("这是xgb")
#     clf = xgb.XGBClassifier(
#         learning_rate=0.05,
#         n_estimators=1000,
#         max_depth=5,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='binary:logistic',
#         nthread=4,
#         seed=27)
#     # 训练
#     clf.fit(train_set_x, train_set_y)
#     #测试
#     pred_value = clf.predict(validate_set)
#     return pred_value
#     # prob = clf.predict_proba(validate_set)
#     # return prob








#
# param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic','eval_metric':'auc'}
# # ,'scale_pos_weight':0.5,'min_child_weight':2,'colsample_bytree':0.3
# param['nthread'] = 4
# watchlist = [(deva, 'eval-00'), (dtrain, 'train')]
# num_round = 2000
# # alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')
# bst = xgb.train(param, dtrain, num_round, watchlist)