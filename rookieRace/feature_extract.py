from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
from com.sun.rookieRace.splitData import *
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import os as os






def get_user_feature(train_feature,train_set):

    train_user = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
    t = train_user[['user_id']]
    t.drop_duplicates(inplace=True)
    t1 = train_user[train_user.date != 'null'][['user_id', 'merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1

    #用户领取优惠券次数
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
    t = pd.merge(t, t1, on='user_id', how='left')
    final_train = pd.merge(train_set, t, on="user_id", how="left")


    t2 = train_user[(train_user.date != 'null') & (train_user.coupon_id != 'null')][['user_id', 'distance']]
    t2 = t2[t2.distance != "null"]
    # t2.replace('null',-1,inplace=True)
    t2.distance = t2.distance.astype('int')
    # t2.replace(-1,np.nan,inplace=True)

    #距离最小值
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)
    final_train = pd.merge(final_train, t3, on="user_id", how="left")


    #距离最大值
    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)
    final_train = pd.merge(final_train, t4, on="user_id", how="left")

    #距离均值
    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)
    final_train = pd.merge(final_train, t5, on="user_id", how="left")


    #距离中位数
    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)
    final_train = pd.merge(final_train, t6, on="user_id", how="left")

    #用户用优惠券的消费次数
    t7 = train_user[(train_user.date != 'null') & (train_user.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()
    final_train = pd.merge(final_train, t7, on="user_id", how="left")


    #用户总的消费数
    t8 = train_user[train_user.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()
    final_train = pd.merge(final_train, t8, on="user_id", how="left")


    #用户领取优惠券的个数
    t9 = train_user[train_user.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()
    final_train = pd.merge(final_train, t9, on="user_id", how="left")



    final_train.count_merchant = final_train.count_merchant.replace(np.nan, 0)
    final_train.buy_use_coupon = final_train.buy_use_coupon.replace(np.nan, 0)
    #用户用优惠券的消费次数占用户总的消费数的比重
    final_train['buy_use_coupon_rate'] = final_train.buy_use_coupon.astype('float') / final_train.buy_total.astype(
        'float')

    #用户用优惠券的消费次数占用户领取个数的比重
    final_train['user_coupon_transfer_rate'] = final_train.buy_use_coupon.astype(
        'float') / final_train.coupon_received.astype('float')

    final_train.buy_total = final_train.buy_total.replace(np.nan, 0)
    final_train.coupon_received = final_train.coupon_received.replace(np.nan, 0)


    #  用户满0~50减的优惠券核销率

                        # 领取并且消费个数
    t10 = train_user[(train_user.coupon_id != "null")&
                     (train_user.date != "null")][["user_id","discount_rate"]]
    t10["discount_man_val"] = t10.discount_rate.apply(get_discount_man_value)
    t10.replace("null", -10, inplace=True)
    t10["discount_man_val"] = t10["discount_man_val"].astype("int64")
    t10 = t10[(t10.discount_man_val > 0)&(t10.discount_man_val < 50)]
    t10 = t10.groupby(["user_id"]).agg("count").reset_index()
    t10.rename(columns={"discount_rate":"count_0_50_man_pay"},inplace=True)
    final_train = pd.merge(final_train, t10, on="user_id", how="left")
    final_train.drop(["discount_man_val"],axis=1,inplace=True)

                        # 领取的个数
    t11 = train_user[(train_user.coupon_id != "null") &
                     (train_user.date == "null")][["user_id", "discount_rate"]]
    t11["discount_man_val"] = t11.discount_rate.apply(get_discount_man_value)
    t11.replace("null", -10, inplace=True)
    t11["discount_man_val"] = t11["discount_man_val"].astype("int64")
    t11 = t11[(t11.discount_man_val > 0) & (t11.discount_man_val < 50)]
    t11 = t11.groupby(["user_id"]).agg("count").reset_index()
    t11.rename(columns={"discount_rate": "count_0_50_man_nopay"}, inplace=True)
    final_train = pd.merge(final_train, t11, on="user_id", how="left")
    final_train.drop(["discount_man_val"], axis=1, inplace=True)

    final_train["0-50pay_nopay_rate"] = final_train.count_0_50_man_pay.astype("float")/\
                                        final_train.count_0_50_man_nopay.astype("float")




    #用户满50~200减的优惠券核销率

                    # 领取并且消费个数
    t12 = train_user[(train_user.coupon_id != "null")&
                     (train_user.date != "null")][["user_id","discount_rate"]]
    t12["discount_man_val"] = t12.discount_rate.apply(get_discount_man_value)
    t12.replace("null", -10, inplace=True)
    t12["discount_man_val"] = t12["discount_man_val"].astype("int64")
    t12 = t12[(t12.discount_man_val >= 50)&(t12.discount_man_val <= 200)]
    # print(t12)
    t12 = t12.groupby(["user_id"]).agg("count").reset_index()

    t12.rename(columns={"discount_rate":"count_50_200_man_pay"},inplace=True)
    final_train = pd.merge(final_train, t12, on="user_id", how="left")
    final_train.drop(["discount_man_val"], axis=1, inplace=True)

                        # 领取的个数
    t13 = train_user[(train_user.coupon_id != "null") &
                     (train_user.date == "null")][["user_id", "discount_rate"]]
    t13["discount_man_val"] = t13.discount_rate.apply(get_discount_man_value)
    t13.replace("null", -10, inplace=True)
    t13["discount_man_val"] = t13["discount_man_val"].astype("int64")
    t13 = t13[(t13.discount_man_val >= 50) & (t13.discount_man_val <= 200)]
    t13 = t13.groupby(["user_id"]).agg("count").reset_index()
    t13.rename(columns={"discount_rate": "count_50_200_man_nopay"}, inplace=True)
    final_train = pd.merge(final_train, t13, on="user_id", how="left")
    final_train.drop(["discount_man_val"], axis=1, inplace=True)

    final_train["50-200pay_nopay_rate"] = final_train.count_50_200_man_pay.astype("float")/\
                                        final_train.count_50_200_man_nopay.astype("float")




    # 用户满200~500减的优惠券核销率

                    # 领取并且消费个数
    t14 = train_user[(train_user.coupon_id != "null") &
                     (train_user.date != "null")][["user_id", "discount_rate"]]
    t14["discount_man_val"] = t14.discount_rate.apply(get_discount_man_value)

    t14.replace("null", -10, inplace=True)
    t14["discount_man_val"] = t14["discount_man_val"].astype("int64")

    t14 = t14[(t14.discount_man_val > 200) & (t14.discount_man_val <= 500)]
    t14 = t14.groupby(["user_id"]).agg("count").reset_index()
    t14.rename(columns={"discount_rate": "count_200_500_man_pay"}, inplace=True)
    final_train = pd.merge(final_train, t14, on="user_id", how="left")
    final_train.drop(["discount_man_val"], axis=1, inplace=True)

                    # 领取的个数
    t15 = train_user[(train_user.coupon_id != "null") &
                     (train_user.date == "null")][["user_id", "discount_rate"]]
    t15["discount_man_val"] = t15.discount_rate.apply(get_discount_man_value)
    t15.replace("null", -10, inplace=True)
    t15["discount_man_val"] = t15["discount_man_val"].astype("int64")

    t15 = t15[(t15.discount_man_val > 200) & (t15.discount_man_val <= 500)]
    t15 = t15.groupby(["user_id"]).agg("count").reset_index()
    t15.rename(columns={"discount_rate": "count_200_500_man_nopay"}, inplace=True)
    final_train = pd.merge(final_train, t15, on="user_id", how="left")
    final_train.drop(["discount_man_val"], axis=1, inplace=True)

    final_train["200-500pay_nopay_rate"] = final_train.count_200_500_man_pay.astype("float") / \
                                          final_train.count_200_500_man_nopay.astype("float")

    # 用户核销优惠券的平均 / 最大 / 最小时间间隔

    t16 = train_user[(train_user.date_received != 'null') &
                     (train_user.date != 'null')][['user_id', 'date_received', 'date']]
    t16['user_pay_date_and_datereceived_gap'] = t16.date + '-' + t16.date_received
    t16['user_pay_date_and_datereceived_gap'] = t16['user_pay_date_and_datereceived_gap'].\
                                                    apply(get_date_date_received_gaps)
    t16 = t16[["user_id","user_pay_date_and_datereceived_gap"]]

    t17 = t16.groupby(["user_id"]).agg("max").reset_index()
    t17.rename(columns={'user_pay_date_and_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    t18 = t16.groupby(["user_id"]).agg("min").reset_index()
    t18.rename(columns={'user_pay_date_and_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)

    t19 = t16.groupby(["user_id"]).agg("mean").reset_index()
    t19.rename(columns={'user_pay_date_and_datereceived_gap': 'mean_user_date_datereceived_gap'}, inplace=True)


    final_train = pd.merge(final_train, t17, on="user_id", how="left")
    final_train = pd.merge(final_train, t18, on="user_id", how="left")
    final_train = pd.merge(final_train, t19, on="user_id", how="left")



    # 用户核销优惠券的平均/最低/最高消费折率
    t20 = train_user[(train_user.date_received != 'null') &
                     (train_user.date != 'null')][['user_id', 'discount_rate']]


    t20["new_coupon_rates"] = t20.discount_rate.apply(get_discount_rate)
    t20 = t20[["user_id","new_coupon_rates"]]

    t21 = t20.groupby(["user_id"]).agg("mean").reset_index()
    t21.rename(columns={'new_coupon_rates': 'mean_user_coupon_rate'}, inplace=True)

    t22 = t20.groupby(["user_id"]).agg("max").reset_index()
    t22.rename(columns={'new_coupon_rates': 'max_user_coupon_rate'}, inplace=True)

    t23 = t20.groupby(["user_id"]).agg("min").reset_index()
    t23.rename(columns={'new_coupon_rates': 'min_user_coupon_rate'}, inplace=True)



    final_train = pd.merge(final_train, t21, on="user_id", how="left")
    final_train = pd.merge(final_train, t22, on="user_id", how="left")
    final_train = pd.merge(final_train, t23, on="user_id", how="left")

    final_train.to_csv("c://aa.csv",index=None)

    # 用户核销满0~50/50~200/200~500减的优惠券占所有核销优惠券的比重

    return final_train


def get_date_date_received_gaps(strs):
    dates,date_received = strs.split('-')
    gap_days = datetime.strptime(dates, '%Y%m%d') - datetime.strptime(date_received, '%Y%m%d')

    return gap_days.days


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


def get_coupon_feature(train_feature,train_set):

    train_coupon = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
    train_set["coupon_id"] = train_set["coupon_id"].astype("str")


    # 用户领取该优惠券次数
    d1 = train_coupon[train_coupon.coupon_id != "null"][["user_id", "coupon_id", "merchant_id"]]
    d1.merchant_id = 1
    d1 = d1.groupby(["user_id", "coupon_id"]).agg("count").reset_index()
    d1.rename(columns={'merchant_id': 'user_get_couponid_count'}, inplace=True)
    final_train = pd.merge(train_set, d1, on=["user_id", "coupon_id"], how="left")


    # 用户消费该优惠券次数
    d2 = train_coupon[(train_coupon.date != 'null')
                      & (train_coupon.date_received != "null")][["user_id", "coupon_id", "merchant_id"]]
    d2.merchant_id = 1
    d2 = d2.groupby(["user_id", "coupon_id"]).agg("count").reset_index()
    d2.rename(columns={"merchant_id": "user_xiaofei_couponid_count"}, inplace=True)
    final_train = pd.merge(final_train, d2, on=["user_id", "coupon_id"], how="left")


    # # 用户对该优惠券的核销率
    final_train.user_xiaofei_couponid_count = final_train.user_xiaofei_couponid_count.replace(np.nan,0)
    final_train.user_get_couponid_count = final_train.user_get_couponid_count.replace(np.nan,0)
    final_train['get_xiaofei_rate'] = final_train.user_xiaofei_couponid_count.astype("float") \
                             / final_train.user_get_couponid_count.astype("float")



    dtemp = train_coupon[train_coupon.coupon_id != "null"][["coupon_id", "discount_rate"]]
    dtemp = dtemp.drop_duplicates()
    # 优惠券折扣率
    dtemp["coupon_discount_rate"] = dtemp.discount_rate.apply(get_discount_rate)
    # 消费券类型(直接优惠为0,满减为1)
    dtemp["coupon_discount_type"] = dtemp.discount_rate.apply(get_coupon_type)
    dtemp.drop(["discount_rate"], inplace=True, axis=1)
    final_train = pd.merge(final_train, dtemp, on=["coupon_id"], how="left")



    # 优惠券出现次数
    d7 = train_coupon[train_coupon.coupon_id != "null"][["coupon_id", "merchant_id"]]
    d7.merchant_id = 2
    d7 = d7.groupby(["coupon_id"]).agg("count").reset_index()
    d7.rename(columns={'merchant_id': 'coupon_count_chuxian'}, inplace=True)
    final_train = pd.merge(final_train, d7, on=["coupon_id"], how="left")



    # 优惠券核销次数
    d8 = train_coupon[(train_coupon.coupon_id != "null") &
                      (train_coupon.date != "null")][["coupon_id", "merchant_id"]]
    d8.merchant_id = 2
    d8 = d8.groupby(["coupon_id"]).agg("count").reset_index()
    d8.rename(columns={'merchant_id': 'coupon_xiaofei_count'}, inplace=True)
    final_train = pd.merge(final_train, d8, on=["coupon_id"], how="left")



    # 优惠券核销率
    final_train['coupon_xiaofei_count'] = final_train['coupon_xiaofei_count'].replace(np.nan,0)
    final_train['coupon_count_chuxian'] = final_train['coupon_count_chuxian'].replace(np.nan, 0)
    final_train["coupon_chuxian_xiaofei_rate"] = final_train.coupon_xiaofei_count.astype("float") / \
                                        final_train.coupon_count_chuxian.astype("float")


    # 满众数,减众数
    d10 = train_coupon[train_coupon.coupon_id != "null"][["coupon_id", "discount_rate"]]  # .drop_duplicates()
    d10["discount_man"] = d10.discount_rate.apply(get_discount_man_value)
    d10["discount_jian"] = d10.discount_rate.apply(get_discount_jian_value)
    d10.drop(["discount_rate"], inplace=True, axis=1)
    d10.drop_duplicates(inplace=True)
    final_train = pd.merge(final_train, d10, on=["coupon_id"], how="left")

    #领取优惠券是一周的第几天
    final_train['day_of_week'] = final_train.date_received.astype('str')\
        .apply(lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)

    #领取优惠券是一月的第几天
    final_train['day_of_month'] = final_train.date_received.astype('str').apply(lambda x: int(x[6:8]))


    return final_train

#



def get_merchant_feature(train_feature,train_set):

    train_merchant = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]



    # 商家优惠券被领取次数
    d = train_merchant[train_merchant.coupon_id != "null"][["merchant_id", "user_id"]]
    d = d.groupby(["merchant_id"]).agg("count").reset_index()
    d.rename(columns={"user_id": "merchant_coupon_lingqued_count"}, inplace=True)
    final_train = pd.merge(train_set, d, on="merchant_id", how="left")


    # 商家优惠券被领取后不核销次数(领取不消费)
    d1 = train_merchant[(train_merchant.coupon_id != "null") & (train_merchant.date == "null")]
    d1 = d1[["merchant_id", "user_id"]]
    d1 = d1.groupby(["merchant_id"]).agg("count").reset_index()
    d1.rename(columns={"user_id": "merchant_coupon_lingqued_noxiaofei_count"}, inplace=True)
    final_train = pd.merge(final_train, d1, on="merchant_id", how="left")



    # 商家优惠券被领取后核销次数(领取消费)
    d2 = train_merchant[(train_merchant.coupon_id != "null") & (train_merchant.date != "null")]
    d2 = d2[["merchant_id", "user_id"]]
    d2 = d2.groupby(["merchant_id"]).agg("count").reset_index()
    d2.rename(columns={"user_id": "merchant_coupon_lingqued_xiaofei_count"}, inplace=True)
    final_train = pd.merge(final_train, d2, on="merchant_id", how="left")


    # 商家优惠券被领取后核销率
    final_train.merchant_coupon_lingqued_xiaofei_count = final_train.merchant_coupon_lingqued_xiaofei_count.replace(np.nan,0)
    final_train.merchant_coupon_lingqued_count = final_train.merchant_coupon_lingqued_count.replace(np.nan,0)

    final_train["merchant_coupon_lingqu_xiaofei_rate"] = final_train.merchant_coupon_lingqued_xiaofei_count.astype("float") / \
                                                         final_train.merchant_coupon_lingqued_count.astype("float")



    # 商家优惠券平均每个用户核销多少张
    d4 = train_merchant[(train_merchant.coupon_id != "null") & (train_merchant.date != "null")]
    d4 = d4[["merchant_id", "user_id", "discount_rate"]]
    d4 = d4.groupby(["merchant_id", "user_id"]).agg("count").reset_index()
    d4.drop(["user_id"], inplace=True, axis=1)
    d4 = d4.groupby(["merchant_id"]).agg("mean").reset_index()
    d4.rename(columns={"discount_rate": "mean_user_xiaofei_count"}, inplace=True)
    final_train = pd.merge(final_train, d4, on="merchant_id", how="left")




    # 商家平均每种优惠券核销多少张
    d5 = train_merchant[(train_merchant.coupon_id != "null") & (train_merchant.date != "null")]
    d5 = d5[["merchant_id", "discount_rate"]]
    d5 = d5.groupby(["merchant_id"]).agg("count").reset_index()
    d5.rename(columns={"discount_rate": "merchant_coupon_mean_xiaofei_count"}, inplace=True)
    final_train = pd.merge(final_train, d5, on="merchant_id", how="left")



    # 商家被核销过的不同优惠券数量
    d6 = train_merchant[(train_merchant.coupon_id != "null") & (train_merchant.date != "null")]
    d6 = d6[["merchant_id", "coupon_id", "discount_rate"]]
    d6 = d6.groupby(["merchant_id", "coupon_id"]).agg("count").reset_index()
    d6.drop(["coupon_id"], inplace=True, axis=1)
    d6 = d6.groupby(["merchant_id"]).agg("count").reset_index()
    d6.rename(columns={"discount_rate": "merchant_xiaofei_notsame_coupon_count"}, inplace=True)
    final_train = pd.merge(final_train, d6, on="merchant_id", how="left")


    # 商家所有被领取过的不同优惠券数量
    d7 = train_merchant[(train_merchant.coupon_id != "null")]
    d7 = d7[["merchant_id", "coupon_id", "discount_rate"]]
    d7 = d7.groupby(["merchant_id", "coupon_id"]).agg("count").reset_index()
    d7.drop(["coupon_id"], inplace=True, axis=1)
    d7 = d7.groupby(["merchant_id"]).agg("count").reset_index()
    d7.rename(columns={"discount_rate": "merchant_xiaofei_notsame_coupon_count_nohexiao"}, inplace=True)
    final_train = pd.merge(final_train, d7, on="merchant_id", how="left")

    final_train.merchant_xiaofei_notsame_coupon_count = final_train.merchant_xiaofei_notsame_coupon_count.replace(np.nan,0)
    final_train.merchant_xiaofei_notsame_coupon_count_nohexiao = final_train.\
                    merchant_xiaofei_notsame_coupon_count_nohexiao.replace(np.nan,0)

    # 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    final_train["nosame_coupon_hexiao_lingqu_rate"] = final_train.merchant_xiaofei_notsame_coupon_count.astype("float") / \
                                                      final_train.merchant_xiaofei_notsame_coupon_count_nohexiao.astype("float")




    # 商家被核销优惠券中的平均/最小/最大用户-商家距离
    d8 = train_merchant[(train_merchant.coupon_id != "null") & (train_merchant.date != "null")][
        ["merchant_id", "distance"]]

    d8["distance"].replace("null", -10, inplace=True)
    d8["distance"] = d8.distance.astype("int")
    d8["distance"].replace(-10, np.nan, inplace=True)

    d9 = d8.groupby('merchant_id').agg('min').reset_index()
    d9.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)
    final_train = pd.merge(final_train, d9, on="merchant_id", how="left")

    d10 = d8.groupby('merchant_id').agg('max').reset_index()
    d10.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)
    final_train = pd.merge(final_train, d10, on="merchant_id", how="left")

    d11 = d8.groupby('merchant_id').agg('mean').reset_index()
    d11.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)
    final_train = pd.merge(final_train, d11, on="merchant_id", how="left")

    d12 = d8.groupby('merchant_id').agg('median').reset_index()
    d12.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)
    final_train = pd.merge(final_train, d12, on="merchant_id", how="left")



    # 商户核销优惠券的平均/最大/最小时间间隔

    d13 = train_merchant[(train_merchant.date_received != 'null') &
                     (train_merchant.date != 'null')][['merchant_id', 'date_received', 'date']]
    d13['merchant_pay_date_and_datereceived_gap'] = d13.date + '-' + d13.date_received
    d13['merchant_pay_date_and_datereceived_gap'] = d13['merchant_pay_date_and_datereceived_gap']. \
        apply(get_date_date_received_gaps)

    d13 = d13[["merchant_id", "merchant_pay_date_and_datereceived_gap"]]

    d14 = d13.groupby(["merchant_id"]).agg("max").reset_index()
    d14.rename(columns={'merchant_pay_date_and_datereceived_gap': 'max_merchant_date_datereceived_gap'}, inplace=True)

    d15 = d13.groupby(["merchant_id"]).agg("min").reset_index()
    d15.rename(columns={'merchant_pay_date_and_datereceived_gap': 'min_merchant_date_datereceived_gap'}, inplace=True)

    d16 = d13.groupby(["merchant_id"]).agg("mean").reset_index()
    d16.rename(columns={'merchant_pay_date_and_datereceived_gap': 'mean_merchant_date_datereceived_gap'}, inplace=True)

    final_train = pd.merge(final_train, d14, on="merchant_id", how="left")
    final_train = pd.merge(final_train, d15, on="merchant_id", how="left")
    final_train = pd.merge(final_train, d16, on="merchant_id", how="left")


    return final_train



def read_data_part_one(train,validate,test):
    change_path_utils(train)
    train_set = pd.read_csv(os.path.basename(train), sep=",")

    change_path_utils(validate)
    validate_set = pd.read_csv(os.path.basename(validate), sep=",")

    change_path_utils(test)
    test_set = pd.read_csv(os.path.basename(test), sep=",")
    return train_set, validate_set, test_set


def get_user_merchant_feature(train_feature,train_set):

    user_merchant = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]


    # 用户领取商家的优惠券次数
    user_merchant = user_merchant[user_merchant.coupon_id != "null"]
    d = user_merchant[["user_id","merchant_id","distance"]]
    d = d.groupby(["user_id","merchant_id"]).agg("count").reset_index()
    d.rename(columns={"distance":"user_get_merchant_coupon_count"},inplace=True)
    final_train = pd.merge(train_set,d,on=["user_id","merchant_id"],how="left")



    # 用户领取商家的优惠券后不核销次数
    d1 = user_merchant[(user_merchant.coupon_id != "null")&
                        (user_merchant.date == "null")][["user_id","merchant_id","distance"]]
    d1 = user_merchant[["user_id", "merchant_id", "distance"]]
    d1 = d1.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d1.rename(columns={"distance": "user_get_merchant_coupon_count_and_nopay"}, inplace=True)
    final_train = pd.merge(final_train, d1, on=["user_id", "merchant_id"], how="left")



    # 用户领取商家的优惠券后核销次数
    d2 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date != "null")][["user_id", "merchant_id", "distance"]]
    d2 = user_merchant[["user_id", "merchant_id", "distance"]]
    d2 = d2.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d2.rename(columns={"distance": "user_get_merchant_coupon_count_and_pay"}, inplace=True)
    final_train = pd.merge(final_train, d2, on=["user_id", "merchant_id"], how="left")
    # 用户领取商家的优惠券后核销率
    final_train["user_get_merchant_pay_rate"] = final_train.user_get_merchant_coupon_count_and_pay.astype("float")/\
                                                final_train.user_get_merchant_coupon_count.astype("float")




                                #  用户总的不核销次数
    d3 = user_merchant[(user_merchant.coupon_id != "null")&
                        (user_merchant.date == "null")][["user_id","distance"]]
    d3 = d3.groupby(["user_id"]).agg("count").reset_index()
    d3.rename(columns={"distance": "user_total_nopay_count"}, inplace=True)
    final_train = pd.merge(final_train,d3, on=["user_id"], how="left")



                            # 用户对每个商家的不核销次数
    d4 = user_merchant[(user_merchant.coupon_id != "null")&
                        (user_merchant.date == "null")][["user_id","merchant_id","distance"]]
    d4 = d4.groupby(["user_id","merchant_id"]).agg("count").reset_index()
    d4.rename(columns={"distance": "user_merchant_nopay_count"}, inplace=True)
    final_train = pd.merge(final_train, d4, on=["user_id","merchant_id"], how="left")


    # 用户对每个商家的不核销次数占用户总的不核销次数的比重
    final_train["user_merchant_and_user_totalnopay_rate"] = final_train.user_merchant_nopay_count.astype("float")/\
                                                            final_train.user_total_nopay_count.astype("float")

    final_train.drop(["user_merchant_nopay_count","user_total_nopay_count"],axis=1,inplace=True)
    # final_train.to_csv("c://bb.csv",index=None)



                            #  用户总的核销次数
    d5 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date != "null")][["user_id", "distance"]]
    d5 = d5.groupby(["user_id"]).agg("count").reset_index()
    d5.rename(columns={"distance": "user_total_pay_count"}, inplace=True)
    final_train = pd.merge(final_train, d5, on=["user_id"], how="left")


                            # 用户对每个商家的核销次数
    d6 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date != "null")][["user_id", "merchant_id", "distance"]]
    d6 = d6.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d6.rename(columns={"distance": "user_merchant_pay_count"}, inplace=True)
    final_train = pd.merge(final_train, d6, on=["user_id","merchant_id"], how="left")

    # 用户对每个商家的优惠券核销次数占用户总的核销次数的比重

    final_train["user_merchant_and_user_totalpay_rate"] = final_train.user_merchant_pay_count.astype("float") / \
                                                            final_train.user_total_pay_count.astype("float")
    final_train.drop(["user_merchant_pay_count", "user_total_pay_count"], axis=1, inplace=True)





                        #商家总的不核销次数
    d7 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date == "null")][["merchant_id", "distance"]]
    d7 = d7.groupby(["merchant_id"]).agg("count").reset_index()
    d7.rename(columns={"distance": "merchant_toal_nopay_count"}, inplace=True)
    final_train = pd.merge(final_train, d7, on=["merchant_id"], how="left")

                        #用户对每个商家的不核销次数
    d8 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date == "null")][["user_id","merchant_id", "distance"]]
    d8 = d8.groupby(["user_id","merchant_id"]).agg("count").reset_index()
    d8.rename(columns={"distance": "user_merchant_nopay_count1"}, inplace=True)
    final_train = pd.merge(final_train, d8, on=["user_id","merchant_id"], how="left")

    #用户对每个商家的不核销次数占商家总的不核销次数的比重
    final_train["user_merchant_and_merchant_totalnopay_rate"] = final_train.user_merchant_nopay_count1.astype("float") / \
                                                          final_train.merchant_toal_nopay_count.astype("float")
    final_train.drop(["user_merchant_nopay_count1", "merchant_toal_nopay_count"], axis=1, inplace=True)




                        # 商家总的核销次数
    d9 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date != "null")][["merchant_id", "distance"]]
    d9 = d9.groupby(["merchant_id"]).agg("count").reset_index()
    d9.rename(columns={"distance": "merchant_toal_pay_count"}, inplace=True)
    final_train = pd.merge(final_train, d9, on=["merchant_id"], how="left")

                        # 用户对每个商家的优惠券核销次数
    d10 = user_merchant[(user_merchant.coupon_id != "null") &
                       (user_merchant.date != "null")][["user_id", "merchant_id", "distance"]]
    d10 = d10.groupby(["user_id", "merchant_id"]).agg("count").reset_index()
    d10.rename(columns={"distance": "user_merchant_pay_count1"}, inplace=True)
    final_train = pd.merge(final_train, d10, on=["user_id", "merchant_id"], how="left")



    # 用户对每个商家的优惠券核销次数占商家总的核销次数的比重
    final_train["user_merchant_and_merchant_totalpay_rate"] = final_train.user_merchant_pay_count1.astype("float") / \
                                                              final_train.merchant_toal_pay_count.astype("float")
    final_train.drop(["user_merchant_pay_count1", "merchant_toal_pay_count"], axis=1, inplace=True)




    # 用户对该商户下的核销优惠券的平均 / 最大 / 最小时间间隔
    d11 = user_merchant[(user_merchant.date_received != 'null') &
                       (user_merchant.date != 'null')][['user_id',"merchant_id",'date_received', 'date']]

    d11['user_merchant_pay_date_and_datereceived_gap'] = d11.date + '-' + d11.date_received
    d11['user_merchant_pay_date_and_datereceived_gap'] = d11['user_merchant_pay_date_and_datereceived_gap']. \
        apply(get_date_date_received_gaps)

    d11 = d11[['user_id',"merchant_id", "user_merchant_pay_date_and_datereceived_gap"]]

    d12 = d11.groupby(['user_id',"merchant_id"]).agg("max").reset_index()
    d12.rename(columns={'user_merchant_pay_date_and_datereceived_gap': 'max_user_merchant_date_datereceived_gap'}, inplace=True)

    d13 = d11.groupby(['user_id',"merchant_id"]).agg("min").reset_index()
    d13.rename(columns={'user_merchant_pay_date_and_datereceived_gap': 'min_user_merchant_date_datereceived_gap'}, inplace=True)

    d14 = d11.groupby(['user_id',"merchant_id"]).agg("mean").reset_index()
    d14.rename(columns={'user_merchant_pay_date_and_datereceived_gap': 'mean_user_merchant_date_datereceived_gap'}, inplace=True)

    final_train = pd.merge(final_train, d12, on=['user_id',"merchant_id"], how="left")
    final_train = pd.merge(final_train, d13, on=['user_id',"merchant_id"], how="left")
    final_train = pd.merge(final_train, d14, on=['user_id',"merchant_id"], how="left")

    return final_train


def get_discount_rate_feature(train_feature,train_set):

    user_coupon = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    # 该折扣率的优惠券被领取的次数
    d = user_coupon[user_coupon.discount_rate != "null"][["discount_rate","distance"]]
    d = d.groupby(["discount_rate"]).agg("count").reset_index()
    d.rename(columns={"distance":"discount_rate_lingqued_count"},inplace=True)
    final_train = pd.merge(train_set, d, on="discount_rate", how="left")


    # 该折扣率的优惠券被领取后并且消费的次数
    d1 = user_coupon[(user_coupon.discount_rate != "null")&(user_coupon.date != "null")][["discount_rate","distance"]]
    d1 = d1.groupby(["discount_rate"]).agg("count").reset_index()
    d1.rename(columns={"distance": "discount_rate_lingqued_count_pay"}, inplace=True)
    final_train = pd.merge(final_train, d1, on="discount_rate", how="left")


    # 该折扣率的优惠券被领取后没有消费的次数
    d2 = user_coupon[(user_coupon.discount_rate != "null") & (user_coupon.date == "null")][
        ["discount_rate", "distance"]]
    d2 = d2.groupby(["discount_rate"]).agg("count").reset_index()
    d2.rename(columns={"distance": "discount_rate_lingqued_count_nopay"}, inplace=True)
    final_train = pd.merge(final_train, d2, on="discount_rate", how="left")


    # 该折扣率所代表的优惠券的消费率
    final_train["discount_rate_lingqued_count_pay_rate"] = final_train.discount_rate_lingqued_count_pay.astype("float")/\
                                                           final_train.discount_rate_lingqued_count.astype("float")

    return final_train


def get_distance_feature(train_feature,train_set):
    distance_form = train_feature[
        ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    print(train_set.dtypes)
    print(train_feature.dtypes)

    # 该距离下领券的次数
    d = distance_form[(distance_form.distance != "null")&
                      (distance_form.coupon_id != "null")][["distance","date"]]

    d = d.groupby(["distance"]).agg("count").reset_index()
    d.rename(columns={"date":"distance_lingqu_coupon_count"},inplace=True)
    final_train = pd.merge(train_set,d,on="distance",how="left")


    # 该距离下领券后消费的次数
    d1 = distance_form[(distance_form.distance != "null") &
                      (distance_form.coupon_id != "null")&(distance_form.date != "null")][["distance", "date"]]
    d1 = d1.groupby(["distance"]).agg("count").reset_index()
    d1.rename(columns={"date": "distance_lingqu_coupon_pay_count"}, inplace=True)
    final_train = pd.merge(final_train, d1, on="distance", how="left")

    # 该距离下的消费率
    final_train["distance_pay_total_rate"] = final_train.distance_lingqu_coupon_pay_count.astype("float")/\
                                             final_train.distance_lingqu_coupon_count.astype("float")

    #该距离下特定商家的领取次数
    d2 = distance_form[(distance_form.distance != "null")&
                      (distance_form.coupon_id != "null")][["distance","merchant_id","date"]]
    d2 = d2.groupby(["distance","merchant_id"]).agg("count").reset_index()
    d2.rename(columns={"date": "distance_merchant_lingqu_coupon_count"}, inplace=True)
    final_train = pd.merge(final_train, d2, on=["distance","merchant_id"], how="left")


    #该距离下特定商家的领取并且消费次数
    d3 = distance_form[(distance_form.distance != "null") &
                       (distance_form.coupon_id != "null") &
                       (distance_form.date != "null")][["distance","merchant_id","date"]]
    d3 = d3.groupby(["distance", "merchant_id"]).agg("count").reset_index()
    d3.rename(columns={"date": "distance_merchant_lingqu_couponpay_count"}, inplace=True)
    final_train = pd.merge(final_train, d3, on=["distance","merchant_id"], how="left")

    #该距离下特定商家的消费率
    final_train["distance_merchant_pay_rate"] = final_train.distance_merchant_lingqu_couponpay_count.astype("float")/\
                                                 final_train.distance_merchant_lingqu_coupon_count.astype("float")

    # 该距离下的特定用户领取并且消费次数
    d4 = distance_form[(distance_form.distance != "null") &
                       (distance_form.coupon_id != "null") &
                       (distance_form.date != "null")][["distance","user_id","date"]]
    d4 = d4.groupby(["distance","user_id"]).agg("count").reset_index()
    d4.rename(columns={"date":"distance_userid_count"},inplace=True)
    final_train = pd.merge(final_train, d4, on=["distance","user_id"], how="left")


    # 该距离下特定用户领取并且消费次数占总的用户的比重
    d5 = distance_form[(distance_form.coupon_id != "null") &
                       (distance_form.date != "null")][["user_id","date"]]

    d5 = d5.groupby(["user_id"]).agg("count").reset_index()
    d5.rename(columns={"date":"total_user_id"},inplace=True)
    final_train = pd.merge(final_train, d5, on=["user_id"], how="left")

    final_train["distance_userid_total_userid_rate"] = final_train.distance_userid_count.astype("float")\
                                                       /final_train.total_user_id.astype("float")


    # 不同距离下核销优惠券的平均 / 最大 / 最小时间间隔

    d6 = distance_form[(distance_form.date_received != 'null') &
                         (distance_form.date != 'null')&
                        (distance_form.distance != "null")][['distance', 'date_received', 'date']]

    d6['distance_pay_date_and_datereceived_gap'] = d6.date + '-' + d6.date_received
    d6['distance_pay_date_and_datereceived_gap'] = d6['distance_pay_date_and_datereceived_gap']. \
        apply(get_date_date_received_gaps)

    d6 = d6[["distance", "distance_pay_date_and_datereceived_gap"]]

    d7 = d6.groupby(["distance"]).agg("max").reset_index()
    d7.rename(columns={'distance_pay_date_and_datereceived_gap': 'max_distance_date_datereceived_gap'}, inplace=True)

    d8 = d6.groupby(["distance"]).agg("min").reset_index()
    d8.rename(columns={'distance_pay_date_and_datereceived_gap': 'min_distance_date_datereceived_gap'}, inplace=True)

    d9 = d6.groupby(["distance"]).agg("mean").reset_index()
    d9.rename(columns={'distance_pay_date_and_datereceived_gap': 'mean_distance_date_datereceived_gap'}, inplace=True)

    final_train = pd.merge(final_train, d7, on="distance", how="left")
    final_train = pd.merge(final_train, d8, on="distance", how="left")
    final_train = pd.merge(final_train, d9, on="distance", how="left")



    return final_train
    # 该距离下特定消费券的消费次数
    # 该距离下消费过特定优惠券数量和总的不同的优惠券数量的比值

def main():
    # 读取之前的特征组成的数据集(训练,验证,测试)
    train_set, validate_set, test_set = read_data_part_one(train_set_pay_path, validate_set_pay_path,
                                                           test_set_pay_path)

    off_train, off_test = loadDataSet()
    train_feature,validate_feature,test_feature = split_data(off_train, off_test)


    #用户相关特征
    user_train = get_user_feature(train_feature,train_set)
    user_validate = get_user_feature(validate_feature, validate_set)
    user_test = get_user_feature(test_feature, test_set)


    # #优惠券相关特征
    # coupon_train = get_coupon_feature(train_feature,user_train)
    # coupon_validate = get_coupon_feature(validate_feature, user_validate)
    # coupon_test = get_coupon_feature(test_feature, user_test)


    #商家相关特征
    merchant_train = get_merchant_feature(train_feature,user_train)
    merchant_validate = get_merchant_feature(validate_feature, user_validate)
    merchant_test = get_merchant_feature(test_feature, user_test)


    #用户-商家相关特征
    user_merchant_train = get_user_merchant_feature(train_feature,merchant_train)
    user_merchant_validate = get_user_merchant_feature(validate_feature, merchant_validate)
    user_merchant_test = get_user_merchant_feature(test_feature, merchant_test)


    # 折扣率相关特征
    discount_train = get_discount_rate_feature(train_feature,user_merchant_train)
    discount_validate = get_discount_rate_feature(validate_feature, user_merchant_validate)
    discount_test = get_discount_rate_feature(test_feature, user_merchant_test)


    #距离相关特征
    final_train = get_distance_feature(train_feature,discount_train)
    final_validate = get_distance_feature(validate_feature,discount_validate)
    final_test = get_distance_feature(test_feature,discount_test)


    #保存结果到本地
    final_train.to_csv('D://数据处理数据源//o2o_m//final_train.csv', index=None)
    final_validate.to_csv('D://数据处理数据源//o2o_m//final_validate.csv', index=None)
    final_test.to_csv('D://数据处理数据源//o2o_m//final_test.csv', index=None)


if __name__ == "__main__":
    main()



















# validate_feature
# validate_coupon = validate_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
#
#
# #用户领取该优惠券次数
# d1 = validate_coupon[validate_coupon.coupon_id != "null"][["user_id","coupon_id","merchant_id"]]
# d1.merchant_id = 1
# d1 = d1.groupby(["user_id","coupon_id"]).agg("count").reset_index()
# d1.rename(columns={'merchant_id':'user_get_couponid_count'},inplace=True)
# # print(d1)
#
# #用户消费该优惠券次数
# d2 = validate_coupon[(validate_coupon.date != 'null')
#                   &(validate_coupon.date_received != "null")][["user_id","coupon_id","merchant_id"]]
#
# d2.merchant_id = 1
# d2 = d2.groupby(["user_id","coupon_id"]).agg("count").reset_index()
# d2.rename(columns={"merchant_id":"user_xiaofei_couponid_count"},inplace=True)
#
# d3 = pd.merge(d1,d2,on=["user_id","coupon_id"],how="inner")
# #用户对该优惠券的核销率
# d3['get_xiaofei_rate'] = d3.user_xiaofei_couponid_count.astype("float")\
#                                    / d3.user_get_couponid_count.astype("float")
#
#
# dtemp = validate_coupon[validate_coupon.coupon_id != "null"][["coupon_id","discount_rate"]]
# dtemp = dtemp.drop_duplicates()
#
# #优惠券折扣率
# dtemp["coupon_discount_rate"] = dtemp.discount_rate.apply(get_discount_rate)
# #消费券类型(直接优惠为0,满减为1)
# dtemp["coupon_discount_type"] = dtemp.discount_rate.apply(get_coupon_type)
# dtemp.drop(["discount_rate"],inplace=True,axis=1)
# # print(dtemp)
#
#
# #优惠券出现次数
# d7 = validate_coupon[validate_coupon.coupon_id != "null"][["coupon_id","merchant_id"]]
# d7.merchant_id = 2
# d7 = d7.groupby(["coupon_id"]).agg("count").reset_index()
# d7.rename(columns={'merchant_id':'coupon_count_chuxian'},inplace=True)
# # print(d7)
#
# #优惠券核销次数
# d8 = validate_coupon[(validate_coupon.coupon_id != "null")&
#                   (validate_coupon.date != "null")][["coupon_id","merchant_id"]]
# d8.merchant_id = 2
# d8 = d8.groupby(["coupon_id"]).agg("count").reset_index()
# d8.rename(columns={'merchant_id':'coupon_xiaofei_count'},inplace=True)
#
# d9 = pd.merge(d7,d8,on="coupon_id",how="right")
# # print(d9)
#
# #优惠券核销率
# d9["coupon_chuxian_xiaofei_rate"] = d9.coupon_xiaofei_count.astype("float") / \
#                                     d9.coupon_count_chuxian.astype("float")
#
# #满众数,减众数
# d10 = validate_coupon[validate_coupon.coupon_id != "null"][["coupon_id","discount_rate"]]#.drop_duplicates()
#
# d10["discount_man"] = d10.discount_rate.apply(get_discount_man_value)
# d10["discount_jian"] = d10.discount_rate.apply(get_discount_jian_value)
# d10.drop(["discount_rate"],inplace=True,axis=1)
# d10.drop_duplicates(inplace=True)
# # print(d10)
#
# validate_coupon_feature = pd.merge(d3,d9,on="coupon_id",how="left")
# validate_coupon_feature = pd.merge(validate_coupon_feature,dtemp,on="coupon_id")
# validate_coupon_feature = pd.merge(validate_coupon_feature,d10,on="coupon_id",how="left")
# validate_coupon_feature.replace("null",0,inplace=True)
# # print(validate_coupon_feature)
# validate_coupon_feature.to_csv("D://数据处理数据源//o2o//validate_coupon_feature.csv",index=None)
#
#
#
# test_feature
# test_coupon = test_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
#
# #用户领取该优惠券次数
# d1 = test_coupon[test_coupon.coupon_id != "null"][["user_id","coupon_id","merchant_id"]]
# d1.merchant_id = 1
# d1 = d1.groupby(["user_id","coupon_id"]).agg("count").reset_index()
# d1.rename(columns={'merchant_id':'user_get_couponid_count'},inplace=True)
# # print(d1)
#
# #用户消费该优惠券次数
# d2 = test_coupon[(test_coupon.date != 'null')
#                   &(test_coupon.date_received != "null")][["user_id","coupon_id","merchant_id"]]
#
# d2.merchant_id = 1
# d2 = d2.groupby(["user_id","coupon_id"]).agg("count").reset_index()
# d2.rename(columns={"merchant_id":"user_xiaofei_couponid_count"},inplace=True)
#
# d3 = pd.merge(d1,d2,on=["user_id","coupon_id"],how="inner")
# #用户对该优惠券的核销率
# d3['get_xiaofei_rate'] = d3.user_xiaofei_couponid_count.astype("float")\
#                                    / d3.user_get_couponid_count.astype("float")
#
#
# dtemp = test_coupon[test_coupon.coupon_id != "null"][["coupon_id","discount_rate"]]
# dtemp = dtemp.drop_duplicates()
#
# #优惠券折扣率
# dtemp["coupon_discount_rate"] = dtemp.discount_rate.apply(get_discount_rate)
# #消费券类型(直接优惠为0,满减为1)
# dtemp["coupon_discount_type"] = dtemp.discount_rate.apply(get_coupon_type)
# dtemp.drop(["discount_rate"],inplace=True,axis=1)
# # print(dtemp)
#
#
# #优惠券出现次数
# d7 = test_coupon[test_coupon.coupon_id != "null"][["coupon_id","merchant_id"]]
# d7.merchant_id = 2
# d7 = d7.groupby(["coupon_id"]).agg("count").reset_index()
# d7.rename(columns={'merchant_id':'coupon_count_chuxian'},inplace=True)
# # print(d7)
#
# #优惠券核销次数
# d8 = test_coupon[(test_coupon.coupon_id != "null")&
#                   (test_coupon.date != "null")][["coupon_id","merchant_id"]]
# d8.merchant_id = 2
# d8 = d8.groupby(["coupon_id"]).agg("count").reset_index()
# d8.rename(columns={'merchant_id':'coupon_xiaofei_count'},inplace=True)
#
# d9 = pd.merge(d7,d8,on="coupon_id",how="right")
# # print(d9)
#
# #优惠券核销率
# d9["coupon_chuxian_xiaofei_rate"] = d9.coupon_xiaofei_count.astype("float") / \
#                                     d9.coupon_count_chuxian.astype("float")
#
# #满众数,减众数
# d10 = test_coupon[test_coupon.coupon_id != "null"][["coupon_id","discount_rate"]]#.drop_duplicates()
#
# d10["discount_man"] = d10.discount_rate.apply(get_discount_man_value)
# d10["discount_jian"] = d10.discount_rate.apply(get_discount_jian_value)
# d10.drop(["discount_rate"],inplace=True,axis=1)
# d10.drop_duplicates(inplace=True)
# # print(d10)
#
# test_coupon_feature = pd.merge(d3,d9,on="coupon_id",how="left")
# test_coupon_feature = pd.merge(test_coupon_feature,dtemp,on="coupon_id")
# test_coupon_feature = pd.merge(test_coupon_feature,d10,on="coupon_id",how="left")
# test_coupon_feature.replace("null",0,inplace=True)
# # print(validate_coupon_feature)
# test_coupon_feature.to_csv("D://数据处理数据源//o2o//test_coupon_feature.csv",index=None)






#
# validate_feature
# validate_merchant = validate_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
#
#
# #商家优惠券被领取次数
# d = validate_merchant[validate_merchant.coupon_id != "null"][["merchant_id","user_id"]]
# d = d.groupby(["merchant_id"]).agg("count").reset_index()
# d.rename(columns={"user_id":"merchant_coupon_lingqued_count"},inplace=True)
# # print(d)
#
# #商家优惠券被领取后不核销次数(领取不消费)
# d1 = validate_merchant[(validate_merchant.coupon_id != "null")&(validate_merchant.date == "null")]
# d1 = d1[["merchant_id","user_id"]]
# d1 = d1.groupby(["merchant_id"]).agg("count").reset_index()
# d1.rename(columns={"user_id":"merchant_coupon_lingqued_noxiaofei_count"},inplace=True)
#
# #商家优惠券被领取后核销次数(领取消费)
# d2 = validate_merchant[(validate_merchant.coupon_id != "null")&(validate_merchant.date != "null")]
# d2 = d2[["merchant_id","user_id"]]
# d2 = d2.groupby(["merchant_id"]).agg("count").reset_index()
# d2.rename(columns={"user_id":"merchant_coupon_lingqued_xiaofei_count"},inplace=True)
#
# # print(d2)
# #商家优惠券被领取后核销率
# d3 = pd.merge(d,d1,on="merchant_id",how="inner")
# d3 = pd.merge(d3,d2,on="merchant_id",how="inner")
# d3["merchant_coupon_lingqu_xiaofei_rate"] = d3.merchant_coupon_lingqued_xiaofei_count.astype("float")/\
#                                             d3.merchant_coupon_lingqued_count.astype("float")
#
# #商家优惠券平均每个用户核销多少张
# d4 = validate_merchant[(validate_merchant.coupon_id != "null")&(validate_merchant.date != "null")]
# d4 = d4[["merchant_id","user_id","discount_rate"]]
# d4 = d4.groupby(["merchant_id","user_id"]).agg("count").reset_index()
# d4.drop(["user_id"],inplace=True,axis=1)
# d4 = d4.groupby(["merchant_id"]).agg("mean").reset_index()
# d4.rename(columns={"discount_rate":"mean_user_xiaofei_count"},inplace=True)
# # print(d4)
#
# #商家平均每种优惠券核销多少张
# d5 = validate_merchant[(validate_merchant.coupon_id != "null")&(validate_merchant.date != "null")]
# d5 = d5[["merchant_id","discount_rate"]]
# d5 = d5.groupby(["merchant_id"]).agg("count").reset_index()
# d5.rename(columns={"discount_rate":"merchant_coupon_mean_xiaofei_count"},inplace=True)
# # print(d5)
#
# #商家被核销过的不同优惠券数量
# d6 = validate_merchant[(validate_merchant.coupon_id != "null")&(validate_merchant.date != "null")]
# d6 = d6[["merchant_id","coupon_id","discount_rate"]]
# d6 = d6.groupby(["merchant_id","coupon_id"]).agg("count").reset_index()
# d6.drop(["coupon_id"],inplace=True,axis=1)
# d6 = d6.groupby(["merchant_id"]).agg("count").reset_index()
# d6.rename(columns={"discount_rate":"merchant_xiaofei_notsame_coupon_count"},inplace=True)
# # print(d6)
#
# #商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
#
# d7 = validate_merchant[(validate_merchant.coupon_id != "null")]
# d7 = d7[["merchant_id","coupon_id","discount_rate"]]
# d7 = d7.groupby(["merchant_id","coupon_id"]).agg("count").reset_index()
# d7.drop(["coupon_id"],inplace=True,axis=1)
# d7 = d7.groupby(["merchant_id"]).agg("count").reset_index()
# d7.rename(columns={"discount_rate":"merchant_xiaofei_notsame_coupon_count_nohexiao"},inplace=True)
#
# d7 = pd.merge(d6,d7,on="merchant_id")
# d7["nosame_coupon_hexiao_lingqu_rate"] = d7.merchant_xiaofei_notsame_coupon_count.astype("float") /\
#                                          d7.merchant_xiaofei_notsame_coupon_count_nohexiao.astype("float")
# # print(d7)
#
# #商家被核销优惠券中的平均/最小/最大用户-商家距离
# d8 = validate_merchant[(validate_merchant.coupon_id != "null")&(validate_merchant.date != "null")][["merchant_id","distance"]]
# d8["distance"].replace("null",-10,inplace=True)
# d8["distance"] = d8.distance.astype("int")
# d8["distance"].replace(-10,np.nan,inplace=True)
#
# d9 = d8.groupby('merchant_id').agg('min').reset_index()
# d9.rename(columns={'distance':'merchant_min_distance'},inplace=True)
#
# d10 = d8.groupby('merchant_id').agg('max').reset_index()
# d10.rename(columns={'distance':'merchant_max_distance'},inplace=True)
#
# d11 = d8.groupby('merchant_id').agg('mean').reset_index()
# d11.rename(columns={'distance':'merchant_mean_distance'},inplace=True)
#
# d12 = d8.groupby('merchant_id').agg('median').reset_index()
# d12.rename(columns={'distance':'merchant_median_distance'},inplace=True)
#
# validate_merchant_feature = pd.merge(d3,d4,on="merchant_id",how="left")
# validate_merchant_feature = pd.merge(validate_merchant_feature,d5,on="merchant_id",how="left")
# validate_merchant_feature = pd.merge(validate_merchant_feature,d7,on="merchant_id",how="left")
# validate_merchant_feature = pd.merge(validate_merchant_feature,d9,on="merchant_id",how="left")
# validate_merchant_feature = pd.merge(validate_merchant_feature,d10,on="merchant_id",how="left")
# validate_merchant_feature = pd.merge(validate_merchant_feature,d11,on="merchant_id",how="left")
# validate_merchant_feature = pd.merge(validate_merchant_feature,d12,on="merchant_id",how="left")
#
# validate_merchant_feature.to_csv("D://数据处理数据源//o2o//validate_merchant_feature.csv",index=None)
#
#
#
#
#
# test_feature
# test_merchant = test_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
#
# #商家优惠券被领取次数
# d = test_merchant[test_merchant.coupon_id != "null"][["merchant_id","user_id"]]
# d = d.groupby(["merchant_id"]).agg("count").reset_index()
# d.rename(columns={"user_id":"merchant_coupon_lingqued_count"},inplace=True)
# # print(d)
#
# #商家优惠券被领取后不核销次数(领取不消费)
# d1 = test_merchant[(test_merchant.coupon_id != "null")&(test_merchant.date == "null")]
# d1 = d1[["merchant_id","user_id"]]
# d1 = d1.groupby(["merchant_id"]).agg("count").reset_index()
# d1.rename(columns={"user_id":"merchant_coupon_lingqued_noxiaofei_count"},inplace=True)
#
# #商家优惠券被领取后核销次数(领取消费)
# d2 = test_merchant[(test_merchant.coupon_id != "null")&(test_merchant.date != "null")]
# d2 = d2[["merchant_id","user_id"]]
# d2 = d2.groupby(["merchant_id"]).agg("count").reset_index()
# d2.rename(columns={"user_id":"merchant_coupon_lingqued_xiaofei_count"},inplace=True)
#
# # print(d2)
# #商家优惠券被领取后核销率
# d3 = pd.merge(d,d1,on="merchant_id",how="inner")
# d3 = pd.merge(d3,d2,on="merchant_id",how="inner")
# d3["merchant_coupon_lingqu_xiaofei_rate"] = d3.merchant_coupon_lingqued_xiaofei_count.astype("float")/\
#                                             d3.merchant_coupon_lingqued_count.astype("float")
#
# #商家优惠券平均每个用户核销多少张
# d4 = test_merchant[(test_merchant.coupon_id != "null")&(test_merchant.date != "null")]
# d4 = d4[["merchant_id","user_id","discount_rate"]]
# d4 = d4.groupby(["merchant_id","user_id"]).agg("count").reset_index()
# d4.drop(["user_id"],inplace=True,axis=1)
# d4 = d4.groupby(["merchant_id"]).agg("mean").reset_index()
# d4.rename(columns={"discount_rate":"mean_user_xiaofei_count"},inplace=True)
# # print(d4)
#
# #商家平均每种优惠券核销多少张
# d5 = test_merchant[(test_merchant.coupon_id != "null")&(test_merchant.date != "null")]
# d5 = d5[["merchant_id","discount_rate"]]
# d5 = d5.groupby(["merchant_id"]).agg("count").reset_index()
# d5.rename(columns={"discount_rate":"merchant_coupon_mean_xiaofei_count"},inplace=True)
# # print(d5)
#
# #商家被核销过的不同优惠券数量
# d6 = test_merchant[(test_merchant.coupon_id != "null")&(test_merchant.date != "null")]
# d6 = d6[["merchant_id","coupon_id","discount_rate"]]
# d6 = d6.groupby(["merchant_id","coupon_id"]).agg("count").reset_index()
# d6.drop(["coupon_id"],inplace=True,axis=1)
# d6 = d6.groupby(["merchant_id"]).agg("count").reset_index()
# d6.rename(columns={"discount_rate":"merchant_xiaofei_notsame_coupon_count"},inplace=True)
# # print(d6)
#
# #商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
#
# d7 = test_merchant[(test_merchant.coupon_id != "null")]
# d7 = d7[["merchant_id","coupon_id","discount_rate"]]
# d7 = d7.groupby(["merchant_id","coupon_id"]).agg("count").reset_index()
# d7.drop(["coupon_id"],inplace=True,axis=1)
# d7 = d7.groupby(["merchant_id"]).agg("count").reset_index()
# d7.rename(columns={"discount_rate":"merchant_xiaofei_notsame_coupon_count_nohexiao"},inplace=True)
#
# d7 = pd.merge(d6,d7,on="merchant_id")
# d7["nosame_coupon_hexiao_lingqu_rate"] = d7.merchant_xiaofei_notsame_coupon_count.astype("float") /\
#                                          d7.merchant_xiaofei_notsame_coupon_count_nohexiao.astype("float")
# # print(d7)
#
# #商家被核销优惠券中的平均/最小/最大用户-商家距离
# d8 = test_merchant[(test_merchant.coupon_id != "null")&(test_merchant.date != "null")][["merchant_id","distance"]]
# d8["distance"].replace("null",-10,inplace=True)
# d8["distance"] = d8.distance.astype("int")
# d8["distance"].replace(-10,np.nan,inplace=True)
#
# d9 = d8.groupby('merchant_id').agg('min').reset_index()
# d9.rename(columns={'distance':'merchant_min_distance'},inplace=True)
#
# d10 = d8.groupby('merchant_id').agg('max').reset_index()
# d10.rename(columns={'distance':'merchant_max_distance'},inplace=True)
#
# d11 = d8.groupby('merchant_id').agg('mean').reset_index()
# d11.rename(columns={'distance':'merchant_mean_distance'},inplace=True)
#
# d12 = d8.groupby('merchant_id').agg('median').reset_index()
# d12.rename(columns={'distance':'merchant_median_distance'},inplace=True)
#
# test_merchant_feature = pd.merge(d3,d4,on="merchant_id",how="left")
# test_merchant_feature = pd.merge(test_merchant_feature,d5,on="merchant_id",how="left")
# test_merchant_feature = pd.merge(test_merchant_feature,d7,on="merchant_id",how="left")
# test_merchant_feature = pd.merge(test_merchant_feature,d9,on="merchant_id",how="left")
# test_merchant_feature = pd.merge(test_merchant_feature,d10,on="merchant_id",how="left")
# test_merchant_feature = pd.merge(test_merchant_feature,d11,on="merchant_id",how="left")
# test_merchant_feature = pd.merge(test_merchant_feature,d12,on="merchant_id",how="left")
#
# test_merchant_feature.to_csv("D://数据处理数据源//o2o//test_merchant_feature.csv",index=None)









#
# # validate_feature
# validate_user = validate_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
#
# t = validate_user[['user_id']]
# t.drop_duplicates(inplace=True)
#
# t1 = validate_user[validate_user.date != 'null'][['user_id', 'merchant_id']]
# t1.drop_duplicates(inplace=True)
# t1.merchant_id = 1
# t1 = t1.groupby('user_id').agg('sum').reset_index()
# t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
#
# t2 = validate_user[(validate_user.date!='null')&(validate_user.coupon_id!='null')][['user_id','distance']]
# # t2.replace('null',-1,inplace=True)
# t2 = t2[t2.distance != "null"]
# t2.distance = t2.distance.astype('int')
# # t2.replace(-1,np.nan,inplace=True)
# t3 = t2.groupby('user_id').agg('min').reset_index()
# t3.rename(columns={'distance':'user_min_distance'},inplace=True)
#
# t4 = t2.groupby('user_id').agg('max').reset_index()
# t4.rename(columns={'distance':'user_max_distance'},inplace=True)
#
# t5 = t2.groupby('user_id').agg('mean').reset_index()
# t5.rename(columns={'distance':'user_mean_distance'},inplace=True)
#
# t6 = t2.groupby('user_id').agg('median').reset_index()
# t6.rename(columns={'distance':'user_median_distance'},inplace=True)
#
# t7 = validate_user[(validate_user.date!='null')&(validate_user.coupon_id!='null')][['user_id']]
# t7['buy_use_coupon'] = 1
# t7 = t7.groupby('user_id').agg('sum').reset_index()
#
# t8 = validate_user[validate_user.date!='null'][['user_id']]
# t8['buy_total'] = 1
# t8 = t8.groupby('user_id').agg('sum').reset_index()
#
# t9 = validate_user[validate_user.coupon_id!='null'][['user_id']]
# t9['coupon_received'] = 1
# t9 = t9.groupby('user_id').agg('sum').reset_index()
#
#
#
# validate_feature = pd.merge(t,t1,on='user_id',how='left')
# validate_feature = pd.merge(validate_feature,t3,on='user_id',how='inner')
# validate_feature = pd.merge(validate_feature,t4,on='user_id',how='left')
# validate_feature = pd.merge(validate_feature,t5,on='user_id',how='left')
# validate_feature = pd.merge(validate_feature,t6,on='user_id',how='left')
# validate_feature = pd.merge(validate_feature,t7,on='user_id',how='left')
# validate_feature = pd.merge(validate_feature,t8,on='user_id',how='left')
# validate_feature = pd.merge(validate_feature,t9,on='user_id',how='left')
#
# validate_feature.count_merchant = validate_feature.count_merchant.replace(np.nan,0)
# validate_feature.buy_use_coupon = validate_feature.buy_use_coupon.replace(np.nan,0)
# validate_feature['buy_use_coupon_rate'] = validate_feature.buy_use_coupon.astype('float') / validate_feature.buy_total.astype('float')
# validate_feature['user_coupon_transfer_rate'] = validate_feature.buy_use_coupon.astype('float') / validate_feature.coupon_received.astype('float')
# validate_feature.buy_total = validate_feature.buy_total.replace(np.nan,0)
# validate_feature.coupon_received = validate_feature.coupon_received.replace(np.nan,0)
# validate_feature.to_csv('D://数据处理数据源//o2o//validate_user_feature.csv',index=None)
#
#
# # test_feature
# test_user = test_feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
# #
# t = test_user[['user_id']]
# t.drop_duplicates(inplace=True)
#
# t1 = test_user[test_user.date != 'null'][['user_id', 'merchant_id']]
# t1.drop_duplicates(inplace=True)
# t1.merchant_id = 1
# t1 = t1.groupby('user_id').agg('sum').reset_index()
# t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
#
# t2 = test_user[(test_user.date!='null')&(test_user.coupon_id!='null')][['user_id','distance']]
# t2 = t2[t2.distance != "null"]
# # t2.replace('null',-1,inplace=True)
# t2.distance = t2.distance.astype('int')
# # t2.replace(-1,np.nan,inplace=True)
# t3 = t2.groupby('user_id').agg('min').reset_index()
# t3.rename(columns={'distance':'user_min_distance'},inplace=True)
#
# print(t3)
# t4 = t2.groupby('user_id').agg('max').reset_index()
# t4.rename(columns={'distance':'user_max_distance'},inplace=True)
#
# t5 = t2.groupby('user_id').agg('mean').reset_index()
# t5.rename(columns={'distance':'user_mean_distance'},inplace=True)
#
# t6 = t2.groupby('user_id').agg('median').reset_index()
# t6.rename(columns={'distance':'user_median_distance'},inplace=True)
#
# t7 = test_user[(test_user.date!='null')&(test_user.coupon_id!='null')][['user_id']]
# t7['buy_use_coupon'] = 1
# t7 = t7.groupby('user_id').agg('sum').reset_index()
#
# t8 = test_user[test_user.date!='null'][['user_id']]
# t8['buy_total'] = 1
# t8 = t8.groupby('user_id').agg('sum').reset_index()
#
# t9 = test_user[test_user.coupon_id!='null'][['user_id']]
# t9['coupon_received'] = 1
# t9 = t9.groupby('user_id').agg('sum').reset_index()
#
#
#
# test_feature = pd.merge(t,t1,on='user_id',how='left')
# test_feature = pd.merge(test_feature,t3,on='user_id',how='inner')
# test_feature = pd.merge(test_feature,t4,on='user_id',how='left')
# test_feature = pd.merge(test_feature,t5,on='user_id',how='left')
# test_feature = pd.merge(test_feature,t6,on='user_id',how='left')
# test_feature = pd.merge(test_feature,t7,on='user_id',how='left')
# test_feature = pd.merge(test_feature,t8,on='user_id',how='left')
# test_feature = pd.merge(test_feature,t9,on='user_id',how='left')
#
# test_feature.count_merchant = test_feature.count_merchant.replace(np.nan,0)
# test_feature.buy_use_coupon = test_feature.buy_use_coupon.replace(np.nan,0)
# test_feature['buy_use_coupon_rate'] = test_feature.buy_use_coupon.astype('float') / test_feature.buy_total.astype('float')
# test_feature['user_coupon_transfer_rate'] = test_feature.buy_use_coupon.astype('float') / test_feature.coupon_received.astype('float')
# test_feature.buy_total = test_feature.buy_total.replace(np.nan,0)
# test_feature.coupon_received = test_feature.coupon_received.replace(np.nan,0)
# test_feature.to_csv('D://数据处理数据源//o2o//test_user_feature.csv',index=None)
# #
#