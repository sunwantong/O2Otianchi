from com.sun.rookieRace.util import *
import pandas as pd

def load_data_set():
    path_1 = "f://resultthree.csv"
    change_path_utils(path_1)
    p1 = pd.read_csv(os.path.basename(path_1), sep=",")

    path_2 = "f://xgb_with_other.csv"
    change_path_utils(path_2)
    p2 = pd.read_csv(os.path.basename(path_2), sep=",",header=None)
    p2.columns = ["user_id","coupon_id","date_received","prob_ling"]

    return p1,p2


def main():
    path_1, path_2 = load_data_set()
    print(path_2)

    new_res = path_1[["user_id", "coupon_id", "date_received"]]

    path_1['rank1'] = path_1['prob'].rank(ascending=False,method="average")

    path_2['rank2'] = path_2['prob_ling'].rank(ascending=False, method="average")


    p1 = path_1["rank1"]
    p2 = path_2["rank2"]


    new_res["rank"] = list(map(lambda x,y: (0.5/x + 0.5/y),p1,p2))
    # print(new_res['rank'])
    new_res.to_csv("c://aa.csv",index=None,header=False)

if __name__ == '__main__':
    main()
