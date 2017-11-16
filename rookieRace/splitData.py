from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
import numpy as np
import pandas as pd
import os as os

"""
 对数据集进行划分
 划分训练集和验证集(线下测试集)
 
 trainDataSet(线下训练集)
 validateDataSet(线下验证集)
 testDataSet(测试集)
 
    (date_received)                              
    testDataSet: 20160701~20160731 (打标好之后的数据), test_feature from 20160315~20160630  (测试集)
    trainDataSet: 20160515~20160615 (打标好之后的数据), train_feature from 20160201~20160514  
    validateDataSet: 20160414~20160514 (打标好之后的数据),validate_feature from 20160101~20160413      
 
"""


def loadDataSet():
    change_path_utils(trainFilePath)
    off_train = pd.read_csv(os.path.basename(trainFilePath),sep=",")
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    change_path_utils(testFilePath)
    off_test = pd.read_csv(os.path.basename(testFilePath), sep=",")
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

    return off_train,off_test

#对数据集进行划分
def split_data(off_train,off_test):
    # testDataSet = off_test
    test_feature = off_train[((off_train.date_received >= '20160501') & (off_train.date_received <= '20160615'))]


    # trainDataSet = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]
    train_feature = off_train[(off_train.date_received >= '20160315') & (off_train.date_received <= '20160501')]


    # validateDataSet = off_train[(off_train.date_received >= '20160401') & (off_train.date_received <= '20160501')]
    validate_feature = off_train[(off_train.date_received >= '20160115') & (off_train.date_received <= '20160315')]


    return train_feature,validate_feature,test_feature


if __name__ == "__main__":
    off_train,off_test = loadDataSet()
    datas = split_data(off_train,off_test)
    print(off_train)
