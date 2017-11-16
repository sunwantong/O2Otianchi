from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
from datetime import *
from pandas import DataFrame,Series
import numpy as np
import pandas as pd
import os as os

def loadDataSet():
    change_path_utils(trainFilePath)
    dataSet = pd.read_csv(os.path.basename(trainFilePath),sep=",")
    return dataSet

#对数据集进行打标处理,生成da
def generateData(dataSet):
    dataSet.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
    dframe = dataSet


    #过滤掉未领取优惠券的
    realDfSet = dframe[dframe['coupon_id'] != 'null']
    #领取优惠券未消费的
    realDfSet['date'] = realDfSet['date'].map(lambda x: "22000120" if x == 'null' else x)

    date = pd.to_datetime(realDfSet['date'])
    date_received = pd.to_datetime(realDfSet['date_received'])

    realDfSet["label"] = list(map(lambda x,y: (x-y).days,date,date_received))
    realDfSet["label"] = realDfSet["label"].map(lambda x:1 if x > 0 and x <=15 else 0)
    realDfSet['date'] = realDfSet['date'].map(lambda x: "null" if x == '22000120' else x)
    realDfSet["label"] = realDfSet["label"].astype("str")
    #返回打标之后的数据集
    # del realDfSet["date"]
    realDfSet.to_csv("D://数据处理数据源//labeled_dataset.csv",index=None)
    return realDfSet



def main():
    dataSet = loadDataSet()
    with_label_Data = generateData(dataSet)
    print(dataSet.dtypes)
    print(with_label_Data.dtypes)

def test():
    dataSet = loadDataSet()
    a = dataSet[(dataSet["Date_received"] >= '20160615')&
                (dataSet["Date_received"] < '20160630')]
    print("记录数:" + str(len(dataSet)))
    print("用户数:" + str(dataSet['User_id'].drop_duplicates().count()))
    print("商户数:" + str(dataSet["Merchant_id"].drop_duplicates().count()))
    print("领券日期统计:", dataSet["Date_received"].value_counts())
    print("领券日期统计dd:", dataSet["Date_received"].value_counts()[:5])
    pass


if __name__ == "__main__":
    main()
