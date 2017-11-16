from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
import numpy as np
import pandas as pd
import os as os




def loadDataSetTrainSet():
    change_path_utils("D://数据处理数据源//o2o//train_user_feature.csv")
    train_user_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//train_user_feature.csv"), sep=",")

    change_path_utils("D://数据处理数据源//o2o//train_merchant_feature.csv")
    train_merchant_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//train_merchant_feature.csv"), sep=",")

    change_path_utils("D://数据处理数据源//o2o//train_coupon_feature.csv")
    train_coupon_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//train_coupon_feature.csv"), sep=",")
    train_coupon_feature.drop(["user_id"],inplace=True,axis=1)
    change_path_utils("D://数据处理数据源//labeled_dataset.csv")
    labeled_data = pd.read_csv(os.path.basename("D://数据处理数据源//labeled_dataset.csv"), sep=",")
    labeled_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received',
                            'label']

    labeled_data = labeled_data[["user_id","merchant_id","coupon_id","date_received","label"]]

    train_dataSet_pred = labeled_data[(labeled_data.date_received >= 20160515) & (labeled_data.date_received <= 20160615)]


    train_set = pd.merge(train_dataSet_pred,train_coupon_feature,on="coupon_id",how="left")
    train_set = pd.merge(train_set, train_user_feature, on="user_id", how="inner")
    train_set = pd.merge(train_set,train_merchant_feature,on="merchant_id",how="left")

    train_set.fillna("0",inplace=True)
    train_set.drop_duplicates(inplace=True)
    print(len(train_set))
    train_set.to_csv("D://数据处理数据源//o2o//train.csv",index=None)


def loadDataSetValidateSet():
    change_path_utils("D://数据处理数据源//o2o//validate_user_feature.csv")
    validate_user_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//validate_user_feature.csv"), sep=",")

    change_path_utils("D://数据处理数据源//o2o//validate_merchant_feature.csv")
    validate_merchant_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//validate_merchant_feature.csv"), sep=",")

    change_path_utils("D://数据处理数据源//o2o//validate_coupon_feature.csv")
    validate_coupon_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//validate_coupon_feature.csv"), sep=",")
    validate_coupon_feature.drop(["user_id"], inplace=True, axis=1)

    change_path_utils("D://数据处理数据源//labeled_dataset.csv")
    labeled_data = pd.read_csv(os.path.basename("D://数据处理数据源//labeled_dataset.csv"), sep=",")
    labeled_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received',
                            'label']

    labeled_data = labeled_data[["user_id", "merchant_id", "coupon_id", "date_received", "label"]]

    validate_dataSet_pred = labeled_data[
        (labeled_data.date_received >= 20160401) & (labeled_data.date_received <= 20160501)]

    validate_set = pd.merge(validate_dataSet_pred, validate_coupon_feature, on="coupon_id", how="left")
    validate_set = pd.merge(validate_set, validate_user_feature, on="user_id", how="inner")
    validate_set = pd.merge(validate_set, validate_merchant_feature, on="merchant_id", how="left")

    validate_set.fillna("0", inplace=True)
    validate_set.drop_duplicates(inplace=True)
    print(len(validate_set))
    validate_set.to_csv("D://数据处理数据源//o2o//validate.csv", index=None)


def loadDataSetTestSet():
    change_path_utils("D://数据处理数据源//o2o//test_user_feature.csv")
    test_user_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//test_user_feature.csv"), sep=",")

    change_path_utils("D://数据处理数据源//o2o//test_merchant_feature.csv")
    test_merchant_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//test_merchant_feature.csv"), sep=",")

    change_path_utils("D://数据处理数据源//o2o//test_coupon_feature.csv")
    test_coupon_feature = pd.read_csv(os.path.basename("D://数据处理数据源//o2o//test_coupon_feature.csv"), sep=",")
    test_coupon_feature.drop(["user_id"], inplace=True, axis=1)

    change_path_utils("D://数据处理数据源//ccf_offline_stage1_test_revised.csv")
    testDataSet = pd.read_csv(os.path.basename("D://数据处理数据源//ccf_offline_stage1_test_revised.csv"), sep=",")
    testDataSet.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

    testDataSet = testDataSet[["user_id", "merchant_id", "coupon_id", "date_received"]]

    test_set = pd.merge(testDataSet, test_coupon_feature, on="coupon_id", how="left")
    test_set = pd.merge(test_set, test_user_feature, on="user_id", how="inner")
    test_set = pd.merge(test_set, test_merchant_feature, on="merchant_id", how="left")

    test_set.fillna("0", inplace=True)
    test_set.drop_duplicates(inplace=True)
    print(len(test_set))
    test_set.to_csv("D://数据处理数据源//o2o//test.csv", index=None)

if __name__ == "__main__":
    loadDataSetTrainSet()
    loadDataSetValidateSet()
    loadDataSetTestSet()