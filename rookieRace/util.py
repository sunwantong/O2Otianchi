import pandas as pd
import numpy as np
import os as os


#切换文件路径
def change_path_utils(path):
    pwd = os.getcwd()
    os.chdir(os.path.dirname(path))