from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
from com.sun.rookieRace.splitData import *
# from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import os as os



"""
 纯消费特征群
"""
def read_data_part_two(train,validate,test):
    change_path_utils(train)
    train_set = pd.read_csv(os.path.basename(train), sep=",")

    change_path_utils(validate)
    validate_set = pd.read_csv(os.path.basename(validate), sep=",")

    change_path_utils(test)
    test_set = pd.read_csv(os.path.basename(test), sep=",")
    return train_set, validate_set, test_set


def split_simple_pay_range(off_train, off_test):


    train_simple_pay_range_feature = off_train[
        (off_train.date >= '20160501') & (off_train.date <= '20160515')]

    validate_simple_pay_range_feature = off_train[
        (off_train.date >= '20160315') & (off_train.date <= '20160401')]

    test_simple_pay_range_feature = off_train[
        (off_train.date >= '20160615') & (off_train.date <= '20160701')]

    return train_simple_pay_range_feature, validate_simple_pay_range_feature, test_simple_pay_range_feature


def get_user_simple_pay_feature(train_feature,train_set):
    train_user = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date']]

    # 用户用优惠券的消费次数
    d = train_user[(train_user.date != 'null') & (train_user.coupon_id != 'null')][['user_id']]
    d['buy_use_coupon_simple_pay'] = 1
    d = d.groupby('user_id').agg('sum').reset_index()
    final_train = pd.merge(train_set, d, on="user_id", how="left")

    # 用户总的消费数
    d1 = train_user[train_user.date != 'null'][['user_id']]
    d1['buy_total_simple_pay'] = 1
    d1 = d1.groupby('user_id').agg('sum').reset_index()
    final_train = pd.merge(final_train, d1, on="user_id", how="left")

    # 用户领取优惠券的个数
    d2 = train_user[train_user.coupon_id != 'null'][['user_id']]
    d2['coupon_received_simple_pay'] = 1
    d2 = d2.groupby('user_id').agg('sum').reset_index()
    final_train = pd.merge(final_train, d2, on="user_id", how="left")



    # 商家优惠券被领取次数
    d3 = train_user[train_user.coupon_id != "null"][["merchant_id", "user_id"]]
    d3 = d3.groupby(["merchant_id"]).agg("count").reset_index()
    d3.rename(columns={"user_id": "merchant_coupon_lingqued_count_spay"}, inplace=True)
    final_train = pd.merge(final_train, d3, on="merchant_id", how="left")

    # 商家优惠券被领取后不核销次数(领取不消费)
    d4 = train_user[(train_user.coupon_id != "null") & (train_user.date == "null")]
    d4 = d4[["merchant_id", "user_id"]]
    d4 = d4.groupby(["merchant_id"]).agg("count").reset_index()
    d4.rename(columns={"user_id": "merchant_coupon_lingqued_noxiaofei_count_spay"}, inplace=True)
    final_train = pd.merge(final_train, d4, on="merchant_id", how="left")

    # 商家优惠券被领取后核销次数(领取消费)
    d5 = train_user[(train_user.coupon_id != "null") & (train_user.date != "null")]
    d5 = d5[["merchant_id", "user_id"]]
    d5 = d5.groupby(["merchant_id"]).agg("count").reset_index()
    d5.rename(columns={"user_id": "merchant_coupon_lingqued_xiaofei_count_spay"}, inplace=True)
    final_train = pd.merge(final_train, d5, on="merchant_id", how="left")

    # 商家优惠券被领取后核销率
    final_train.merchant_coupon_lingqued_xiaofei_count_spay = final_train.merchant_coupon_lingqued_xiaofei_count_spay.replace(
        np.nan, 0)
    final_train.merchant_coupon_lingqued_count_spay = final_train.merchant_coupon_lingqued_count_spay.replace(np.nan, 0)

    final_train["merchant_coupon_lingqu_xiaofei_rate_spay"] = final_train.merchant_coupon_lingqued_xiaofei_count_spay.astype(
        "float") / \
                                                         final_train.merchant_coupon_lingqued_count_spay.astype("float")

    # 用户领取商家的优惠券次数
    train_user = train_user[train_user.coupon_id != "null"]
    d6 = train_user[["user_id", "merchant_id", "distance"]]
    d6 = d6.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d6.rename(columns={"distance": "user_get_merchant_coupon_count_spay"}, inplace=True)
    final_train = pd.merge(final_train, d6, on=["user_id", "merchant_id"], how="left")

    # 用户领取商家的优惠券后不核销次数
    d7 = train_user[(train_user.coupon_id != "null") &
                       (train_user.date == "null")][["user_id", "merchant_id", "distance"]]
    d7 = train_user[["user_id", "merchant_id", "distance"]]
    d7 = d7.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d7.rename(columns={"distance": "user_get_merchant_coupon_count_and_nopay_spay"}, inplace=True)
    final_train = pd.merge(final_train, d7, on=["user_id", "merchant_id"], how="left")

    # 用户领取商家的优惠券后核销次数
    d8 = train_user[(train_user.coupon_id != "null") &
                       (train_user.date != "null")][["user_id", "merchant_id", "distance"]]
    d8 = train_user[["user_id", "merchant_id", "distance"]]
    d8 = d8.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d8.rename(columns={"distance": "user_get_merchant_coupon_count_and_pay_spay"}, inplace=True)
    final_train = pd.merge(final_train, d8, on=["user_id", "merchant_id"], how="left")


    # 用户领取商家的优惠券后核销率
    final_train["user_get_merchant_pay_rate_spay_spay"] = final_train.user_get_merchant_coupon_count_and_pay_spay.astype("float") / \
                                                final_train.user_get_merchant_coupon_count_spay.astype("float")

    return final_train

def main():
    train_set, validate_set, test_set = read_data_part_two(train_set_label_path, validate_set_label_path,
                                                           test_set_label_path)
    off_train, off_test = loadDataSet()
    train_feature, validate_feature, test_feature = split_simple_pay_range(off_train, off_test)

    #用户特征
    final_train = get_user_simple_pay_feature(train_feature,train_set)
    final_validate = get_user_simple_pay_feature(validate_feature, validate_set)
    final_test = get_user_simple_pay_feature(test_feature, test_set)

    final_train.to_csv("D://数据处理数据源//o2osimplepay//train.csv",index=None)
    final_validate.to_csv("D://数据处理数据源//o2osimplepay//validate.csv", index=None)
    final_test.to_csv("D://数据处理数据源//o2osimplepay//test.csv", index=None)



if __name__ == "__main__":
    main()