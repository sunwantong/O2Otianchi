

"""
  数据集路径
"""


#原始训练集(未打标)
trainFilePath = "E://数据处理数据源//ccf_offline_stage1_train.csv"
#线上测试集
testFilePath = "E://数据处理数据源//ccf_offline_stage1_test_revised.csv"
#对线下训练集进行打标之后的结果
labeled_train_Data = "E://数据处理数据源//labeled_dataset.csv"

#特征数据集
test_feature_path = 'E://数据处理数据源//o2o//test_feature.csv'
validate_feature_path = 'E://数据处理数据源//o2o//validate_feature.csv'
train_feature_path = 'E://数据处理数据源//o2o//train_feature.csv'


# #训练集，验证集，测试集(0.7)(feature_extract里边)[存储标签]
train_set_label_path = "E://数据处理数据源//o2olabel//train.csv"
validate_set_label_path = "E://数据处理数据源//o2olabel//validate.csv"
test_set_label_path = "E://数据处理数据源//o2olabel//test.csv"


#训练集，验证集，测试集(0.7)(feature_extract里边)[存储只消费]
train_set_pay_path = "E://数据处理数据源//o2osimplepay//train.csv"
validate_set_pay_path = "E://数据处理数据源//o2osimplepay//validate.csv"
test_set_pay_path = "E://数据处理数据源//o2osimplepay//test.csv"


#最终的训练集，验证集，测试集(main里边)
final_train_set = "E://数据处理数据源//o2o_m//final_train.csv"
final_validate_set = "E://数据处理数据源//o2o_m//final_validate.csv"
final_test_set = "E://数据处理数据源//o2o_m//final_test.csv"