from pandas import DataFrame,Series
from com.sun.rookieRace.config import *
from com.sun.rookieRace.util import *
from com.sun.rookieRace.splitData import *
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import os as os


"""
模型融合
 xgb_set.drop_duplicates(["user_id","coupon_id","date_received"],keep="first",inplace=True)
    xgb_set.sort_values(by=['prob'], ascending=True, inplace=True)
"""



def load_data_set():
    rf = "f://resultrf.csv"
    change_path_utils(rf)
    rf_set = pd.read_csv(os.path.basename(rf), sep=",")

    xgbs = "f://resultthree.csv"
    change_path_utils(xgbs)
    xgb_set = pd.read_csv(os.path.basename(xgbs), sep=",")

    gbtree = "f://resultgbtree.csv"
    change_path_utils(gbtree)
    gbtree_set = pd.read_csv(os.path.basename(gbtree), sep=",")

    other_set = "f://other//xgb_chen.csv"
    change_path_utils(other_set)
    other_set = pd.read_csv(os.path.basename(other_set), sep=",")

    gbtree_set.drop(["prob_pos"],inplace=True,axis=1)
    return rf_set,xgb_set,gbtree_set,other_set


def model_fuse(rf_set,xgb_set,gbtree_set,other_set):
    new_res = xgb_set[["user_id","coupon_id","date_received"]]

    ########rf-xgbs########
    # rf = rf_set["prob_neg"]
    # xgbs = xgb_set["prob"]
    # new_res["prob"] = list(map(lambda x,y: (x*0.3+y*0.7),rf,xgbs))



    #########gbtree-xgbs####
    # gbtree = gbtree_set["prob_neg"]
    # xgbs = xgb_set["prob"]
    # new_res["prob"] = list(map(lambda x, y: (x * 0.35 + y * 0.65), gbtree, xgbs))



    #########gbtree-xgbs-rf#####
    # gbtree = gbtree_set["prob_neg"]
    # xgbs = xgb_set["prob"]
    # rf = rf_set["prob_neg"]
    # new_res["prob"] = list(map(lambda x, y,z: (x * 0.1 + y * 0.3+z*0.6), rf,gbtree, xgbs))


    #########fuse-with-other
    other = other_set["prob"]
    xgbs = xgb_set["prob"]

    new_res["prob"] = list(map(lambda x, y: (x * 0.5 + y * 0.5), other,xgbs))

    return new_res


def main():
    rf_set, xgb_set,gbtree_set,other_set = load_data_set()
    result = model_fuse(rf_set,xgb_set,gbtree_set,other_set)
    result.to_csv("f://xgb_with_other.csv",index=None,header=False)

if __name__ == "__main__":
    main()