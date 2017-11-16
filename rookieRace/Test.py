from datetime import *
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import os as os
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime
from datetime import date
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
def getTest():
    data1 = {"lkey": ['b', 'b', 'a', 'c', 'a', 'a', 'b'], "data1": ["1", "1", "3", "4", "5", "6", "7", ]}
    data2 = {"rkey": ['a', 'b', 'd'], "data2": range(3)}
    df2 = pd.DataFrame(data2, columns=["rkey", "data2", "dd"], index=["ni", "hao", "ma"])
    df1 = pd.DataFrame(data1)
    print(df1)
    df1.drop_duplicates(inplace=True)
    print(df1)


def testIloc():
    testPivot()


def testPivot():
    labeled_train_Data = "D://数据处理数据源//labeled_dataset.csv"
    pwd = os.getcwd()
    os.chdir(os.path.dirname(labeled_train_Data))
    dataSet = pd.read_csv(os.path.basename(labeled_train_Data), sep=",")
    df = dataSet.head(10)
    df = DataFrame(df)
    # print(df)
    dfNew = df.iloc[:4, :3]
    # dfnews = dfNew.copy()
    # dfnews["wo"] = ["a","b","c","d"]
    print(dfNew)
    # c = dfNew .apply(lambda x: x.max()-x.min())
    # print(dfNew.groupby("User_id").sum())
    # print(pd.pivot_table(dfNew, index="User_id", aggfunc=np.sum, values="Coupon_id"))
    # print(dfNew["User_id"].dtype)
    # dfNew["User_id"] = dfNew["User_id"].astype("float64")
    # print(dfNew["User_id"].dtype)
    # df = df.set_index(["User_id","Merchant_id"])
    # print(df)
    dfNew.drop(["Coupon_id"],inplace=True,axis=1)
    print(dfNew)


def testPlot():
    labeled_train_Data = "D://数据处理数据源//ccf_offline_stage1_train.csv"
    pwd = os.getcwd()
    os.chdir(os.path.dirname(labeled_train_Data))
    dataSet = pd.read_csv(os.path.basename(labeled_train_Data), sep=",")
    df = dataSet
    df = DataFrame(df)
    df["Discount_rate"].value_counts().plot(kind="bar",alpha=0.9,title="aa")
    plt.show()

def testGroupBy():
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                             'foo', 'bar', 'foo', 'foo'],
                       'B': ['null', 'one', 'null', 'three',
                             'two', 'two', 'one', 'one'],
                       # 'C': np.random.randn(8),
                       'c': ['null', 'on ', 'null', 'thre ',
                             'tw ', 't ', 'one', 'on ']
                      })
    df3 = df.groupby(['A',"B"])['c'].agg(lambda x: ':'.join(x)).reset_index()
    # print(df3)
    # pass

    # df["A"].drop_duplicates(inplace=True)
    # df3 = df.groupby(['A',"B"]).agg("count").reset_index()
    # print(df)
    # print(d)
    print(df3)

def xgbs(train_set_x, train_set_y, validate_set):
    clf = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        seed=27)
    # 训练
    clf.fit(train_set_x, train_set_y)
    pred_value = clf.predict(validate_set.values)

    return pred_value

def testdatetime():
    s1 = "20160529"
    s2 = "20160524"
    time1 = datetime.strptime(s1, '%Y%m%d')
    time2 = datetime.strptime(s2, '%Y%m%d')
    d = (time1 - time2).days
    print(d)
    # nums = [[7,2,4],[5,6,np.nan],[6,1,5],[7,2,8],[7,2,20]]
    # df = pd.DataFrame(nums,columns=["aa","bb","cc"])
    # df_bak = df
    # c = df.groupby(["aa","bb"]).agg(np.size).reset_index()
    # # c = df.groupby(["aa", "bb"]).["cc"].

    #
    # print(df)
    # print(c)


def test_union():
    change_path_utils(trainFilePath)
    train_set = pd.read_csv(os.path.basename(trainFilePath), sep=",")
    train_set.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']


    #用户领券个数
    d = train_set[(train_set.coupon_id != "null")]
    d = d[(d.date > "20160220")&
                    (d.date < "20160520")][["user_id","date_received"]]

    d = d.groupby(["user_id"]).agg("count").reset_index()
    # d.rename(columns={"date_received":"get_count"},inplace=True)

    #用户领券并且消费个数
    d1 = train_set[(train_set.coupon_id != "null")&(train_set.date != "null")]
    d1 = d1[(d1.date > "20160220") &
          (d1.date < "20160520")][["user_id", "date"]]

    d1 = d1.groupby(["user_id"]).agg("count").reset_index()
    # d1.rename(columns={"date": "pay_count"}, inplace=True)



    final_set = pd.merge(d,d1,on="user_id",how="left")
    final_set.set_index("user_id",inplace=True)

    final_set.plot.bar()
    plt.show()
    print(final_set)







    # change_path_utils(testFilePath)
    # test_set = pd.read_csv(os.path.basename(testFilePath), sep=",")

    # # aa = set(train_set["Merchant_id"])  #&
    # # bb = set(test_set["Merchant_id"])
    # # print(len(aa))
    # # print(len(bb))
    # # cc = len(aa and bb)
    # # print(cc)
    # a = [1,2,4,5,6]
    # b = [4,5,6,7,9]
    # print(a and b)

def test_list_union():
    a = [1,3,4,6]
    b = [5,7,4,6]
    #交集
    res_jiao = set(a).intersection(set(b))
    #并集
    res_bing = set(a).union(set(b))
    #差集
    res_cha = set(a).difference(set(b))
    print(list(res_cha))
    pass


#测试列表生成式
def test_list_generator():
    nums = list(range(1,20))
    print(nums)
    a = [ele for ele in nums if ele % 2 == 0]
    # b = nums.map(lambda x: x if x > 10 else -x)
    print(a)

if __name__ == "__main__":
    # getTest()o
    # testIloc()
    # testPlot()
    # testGroupBy()
    # test_xgboost()
    # testdatetime()
    # test_union()
    # test_list_generator()
    test_list_union()