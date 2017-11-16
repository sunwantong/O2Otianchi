from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
from com.sun.rookieRace.splitData import *
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import os as os


def split_pred_range_feature(off_train, off_test):
    # testDataSet = off_test
    test_pred_range_feature = off_test

    # trainDataSet = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]
    train_pred_range_feature = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]

    # validateDataSet = off_train[(off_train.date_received >= '20160401') & (off_train.date_received <= '20160501')]
    validate_pred_range_feature = off_train[(off_train.date_received >= '20160401') & (off_train.date_received <= '20160501')]

    return train_pred_range_feature, validate_pred_range_feature,test_pred_range_feature


def get_day_gap_before(strs):
    date_received, dates = strs.split('-')
    dates = dates.split(':')
    gaps = []
    date_received = str(date_received)
    # print(date_received)
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.strptime(date_received,'%Y%m%d') - datetime.strptime(dt,'%Y%m%d')
        gap_days = gap_days.days
        if gap_days > 0:
            gaps.append(gap_days)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(strs):
    date_received, dates = strs.split('-')
    dates = dates.split(':')
    date_received = str(date_received)
    gaps = []
    print()
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.strptime(dt, '%Y%m%d') - datetime.strptime(date_received, '%Y%m%d')
        gap_days = gap_days.days
        if gap_days > 0:
            gaps.append(gap_days)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)

"""
是：1
否：0
"""

def is_first_get_coupon(strs):
    date_received, dates = strs.split("-")
    dates = dates.split(':')
    date_received = str(date_received)
    gaps = []
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.strptime(dt, '%Y%m%d') - datetime.strptime(date_received, '%Y%m%d')
        gap_days = gap_days.days
        if gap_days < 0:
            return 0  # 不是第一次
        gaps.append(gap_days)
    if len(gaps) == 1:  # 是第一次，也是最后一次
        return 1
    return 1

"""
是最后一次：1
不是最后一次：0
"""

def is_last_get_coupon(strs):
    date_received, dates = strs.split("-")
    dates = dates.split(':')
    date_received = str(date_received)
    gaps = []
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.strptime(date_received, '%Y%m%d') - datetime.strptime(dt, '%Y%m%d')
        gap_days = gap_days.days
        if gap_days < 0:
            return 0  # 不是最后一次
        gaps.append(gap_days)
    if len(gaps) == 1:  # 是第一次，也是最后一次
        return 1
    return 1



def get_discount_man_value(strs):
    s = str(strs)
    s = s.split(":")
    if len(s) == 1:
        return "null"
    else:
        return s[0]

def get_discount_jian_value(strs):
    s = str(strs)
    s = s.split(":")
    if len(s) == 1:
        return "null"
    else:
        return s[1]


def get_date_date_received_gaps(strs):
    dates,date_received = strs.split('-')
    gap_days = datetime.strptime(dates, '%Y%m%d') - datetime.strptime(date_received, '%Y%m%d')

    return gap_days.days

#优惠券折扣率
def get_discount_rate(strs):
    s = str(strs)
    s = s.split(":")
    if len(s) == 1:
        return float(s[0])
    else:
        return 1 - float(s[1]) / float(s[0])


#判断优惠券类型(直接优惠为0,满减为1)
def get_coupon_type(strs):
    s = str(strs)
    s = s.split(":")
    if len(s) == 1:
        return 0
    else:
        return 1


def get_label_feature(train_feature,train_dataSet_pred):

    train_other = train_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']]


    # 用户领取的所有优惠券的数目
    d = train_other[["user_id", "distance"]]
    d = d.groupby(["user_id"]).agg("count").reset_index()
    d.rename(columns={"distance": "user_get_total_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d, on="user_id", how="left")



    # 用户领取的特定优惠券数目
    d1 = train_other[train_other.coupon_id != "null"][["user_id", "coupon_id", "distance"]]
    d1 = d1.groupby(["user_id", "coupon_id"]).agg("count").reset_index()
    d1.rename(columns={"distance": "user_teding_coupon_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d1, on=["user_id", "coupon_id"], how="left")



    # 用户领取特定商家的优惠券数目
    d2 = train_other[["user_id", "merchant_id", "distance"]]
    d2 = d2.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d2.rename(columns={"distance": "user_merchant_couponcount"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d2, on=["user_id", "merchant_id"], how="left")



    # 用户领取的不同商家数目
    d3 = d2.drop(["user_merchant_couponcount"], axis=1)
    d3 = d3.groupby(["user_id"]).agg("count").reset_index()
    d3.rename(columns={"merchant_id": "user_get_nosame_merchant"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d3, on=["user_id"], how="left")



    # 用户领取的所有优惠券种类数目
    d4 = d1.drop(["user_teding_coupon_count"], axis=1)
    d4 = d4.groupby(["user_id"]).agg("count").reset_index()
    d4.rename(columns={"coupon_id": "user_get_all_couponzhonglei_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d4, on=["user_id"], how="left")



    # 商家被领取的优惠券数目
    d5 = train_other[["merchant_id", "distance"]]
    d5 = d5.groupby(["merchant_id"]).agg("count").reset_index()
    d5.rename(columns={"distance": "merchant_lingqued_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d5, on=["merchant_id"], how="left")



    # 商家被领取的特定优惠券数目
    d6 = train_other[["merchant_id", "coupon_id", "distance"]]
    d6 = d6.groupby(["merchant_id", "coupon_id"]).agg("count").reset_index()
    d6.rename(columns={"distance": "merchant_lingqu_tedingcoupon_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d6, on=["merchant_id", "coupon_id"], how="left")



    # 商家被多少不同用户领取的数目
    d7 = d2.drop(["user_merchant_couponcount"], axis=1)
    d7 = d7.groupby(["merchant_id"]).agg("count").reset_index()
    d7.rename(columns={"user_id": "merchant_lingqu_nosame_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d7, on=["merchant_id"], how="left")
    # print(train_dataSet_pred)


    # 商家发行的所有优惠券种类数目
    d8 = d6.drop(["merchant_lingqu_tedingcoupon_count"], axis=1)
    d8 = d8.groupby(["merchant_id"]).agg("count").reset_index()
    d8.rename(columns={"coupon_id": "merchant_faxing_allcoupon_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d8, on=["merchant_id"], how="left")



    # 用户当天领取的优惠券数目
    d9 = train_other[["user_id", "date_received", "distance"]]
    d9 = d9.groupby(["user_id", "date_received"]).agg("count").reset_index()
    d9.rename(columns={"distance": "user_dangtian_lingqu_coupon_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d9, on=["user_id", "date_received"], how="left")



    # 用户当天领取的特定优惠券数目
    d10 = train_other[["user_id", "date_received", "coupon_id", "distance"]]
    d10 = d10.groupby(["user_id", "date_received", "coupon_id"]).agg("count").reset_index()
    d10.rename(columns={"distance": "user_dangtian_lingqu_tedingcoupon_count"}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d10, on=["user_id", "date_received", "coupon_id"], how="left")



    # 用户上/下一次领取的时间间隔
    d11 = train_other[['user_id', 'coupon_id', 'date_received']]
    d11.date_received = d11.date_received.astype('str')
    d11 = d11.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    d11.rename(columns={'date_received': 'dates'}, inplace=True)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d11, on=["user_id", "coupon_id"], how="left")


    train_dataSet_pred['date_received_and_dates'] = train_dataSet_pred.date_received.astype('str') + '-' + \
                                                    train_dataSet_pred.dates

    train_dataSet_pred['user_before_day_lingqucoupon_gap'] = train_dataSet_pred. \
        date_received_and_dates.apply(get_day_gap_before)

    train_dataSet_pred['user_after_day_lingqucoupon_gap'] = train_dataSet_pred. \
        date_received_and_dates.apply(get_day_gap_after)



    #领券日期是一周的第几天
    train_dataSet_pred['day_of_week_label_range'] = train_dataSet_pred.date_received.astype('str') \
        .apply(lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)




    # 领取优惠券是一月的第几天
    train_dataSet_pred['day_of_month_label_range'] = train_dataSet_pred.date_received.astype('str').apply(lambda x: int(x[6:8]))



    d12 = train_other[train_other.coupon_id != "null"][["coupon_id", "discount_rate"]]
    d12 = d12.drop_duplicates()
    # 优惠券折扣率
    d12["coupon_discount_rate_label_range"] = d12.discount_rate.apply(get_discount_rate)



    # 消费券类型(直接优惠为0,满减为1)
    d12["coupon_discount_type_label_range"] = d12.discount_rate.apply(get_coupon_type)
    # d12.drop(["discount_rate"], inplace=True, axis=1)
    train_dataSet_pred = pd.merge(train_dataSet_pred, d12, on=["coupon_id"], how="left")



    # 用户是否是第一次领取特定优惠券
    train_dataSet_pred["is_first_get_coupon"] = train_dataSet_pred.date_received_and_dates.apply(is_first_get_coupon)



    # 用户是否是最后一次领取特定优惠券
    train_dataSet_pred["is_last_get_coupon"] = train_dataSet_pred.date_received_and_dates.apply(is_last_get_coupon)



    d13 = train_other[['user_id', 'coupon_id', 'date_received']]
    d13.date_received = d13.date_received.astype('str')
    d13 = d13.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    d13['lingqu_number'] = d13.date_received.apply(lambda s: len(s.split(':')))
    d13.rename(columns={"date_received":"dates_more_received"},inplace=True)
    d13 = d13[d13.lingqu_number > 1]

    d13['max_date_received'] = d13.dates_more_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    d13['min_date_received'] = d13.dates_more_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    d13 = d13[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]


    train_dataSet_pred = pd.merge(train_dataSet_pred, d13, on=["user_id", "coupon_id"], how="left")
    train_dataSet_pred['datereceided_and_max_received_gap'] = train_dataSet_pred.max_date_received - \
                                                              train_dataSet_pred.date_received.astype("float")

    train_dataSet_pred['datereceided_and_min_received_gap'] = train_dataSet_pred.date_received.astype("float") - \
                                                              train_dataSet_pred.min_date_received

    # 满减的减的额度
    train_dataSet_pred["discount_jian_label_range"] = train_dataSet_pred.discount_rate.apply(get_discount_jian_value)
    #满减的满的额度
    train_dataSet_pred["discount_man_label_range"] = train_dataSet_pred.discount_rate.apply(get_discount_man_value)

    # 满减的最低消费
    # train_dataSet_pred["man_jain_min_pay"] =


    return train_dataSet_pred






def main():
    off_train, off_test = loadDataSet()
    train_feature,validate_feature,test_feature = split_pred_range_feature(off_train, off_test)
    test_feature["coupon_id"] = test_feature["coupon_id"].astype("str")

    # 打完标之后的数据
    labeled_data = pd.read_csv(os.path.basename("D://数据处理数据源//labeled_dataset.csv"), sep=",")
    labeled_data = labeled_data[["user_id", "merchant_id", "coupon_id", "date_received", "distance","label"]]
    labeled_data["coupon_id"] = labeled_data["coupon_id"].astype("str")
    labeled_data["date_received"] = labeled_data["date_received"].astype("str")
    labeled_data["label"] = labeled_data["label"].astype("str")

    # 训练
    train_dataSet_pred = labeled_data[
        (labeled_data.date_received >= '20160515') & (labeled_data.date_received <= '20160615')]

    # 验证
    validate_dataSet_pred = labeled_data[
        (labeled_data.date_received >= '20160401') & (labeled_data.date_received <= '20160501')]

    # 测试
    test_dataset_pred = off_test[["user_id", "merchant_id", "coupon_id", "date_received","distance"]]

    test_dataset_pred["coupon_id"] = test_dataset_pred["coupon_id"].astype("str")
    test_dataset_pred["date_received"] = test_dataset_pred["date_received"].astype("str")

    train = get_label_feature(train_feature,train_dataSet_pred)
    validate = get_label_feature(validate_feature, validate_dataSet_pred)
    test = get_label_feature(test_feature, test_dataset_pred)


    #保存结果
    train.to_csv("D://数据处理数据源//o2olabel//train.csv", index=None)
    validate.to_csv("D://数据处理数据源//o2olabel//validate.csv", index=None)
    test.to_csv("D://数据处理数据源//o2olabel//test.csv", index=None)



if __name__ == "__main__":
    main()
