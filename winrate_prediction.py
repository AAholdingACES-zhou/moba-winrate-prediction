#导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#train 数据集的维度
train_df.shape

#提取train 数据集中的因变量y
train_df_copy = train_df.copy()
label = train_df.pop('y')
#把train_df 和 test_df 拼接，可以同时进行数据清洗
df = pd.concat([train_df, test_df], keys=['train', 'test'])
df.shape

#数据清洗
#检查空值存在的列
columns_isull = df.columns[df.isnull().sum() != 0]
columns_isull

#检查每局比赛的数据，两队的英雄数量是否正确（都应该等于5）
df['my_heros'] = 0
df['enemy_heros'] = 0
for index, row in df.iterrows():
row_list = row.to_list()[3:-2]
df.loc[index, 'my_heros'] = row_list.count(1) #检查我方英雄数量
df.loc[index, 'enemy_heros'] = row_list.count(-1) #检查敌方的英雄数量


df[(df['my_heros']!= 5) | (df['enemy_heros']!= 5)] #检查后发现没有不合规的数据

#故此时可以确定应该将缺失值填补为0，并将异常值100更改为0

#填补空缺值为0
df.fillna(0, inplace=True)

#修改异常数据为0
rows, cols = np.where(df.values==100)
for index, value in enumerate(rows):
df.iloc[value, cols[index]] = 0

#之后可以查看从未被选择过的英雄，这些英雄对于胜率预测的参考价值很小，故可以被去除。

#检查是否存在英雄从来没有被选择过
delete_col = []
for col in df.columns:
col_0 = df[col].to_list().count(0)
if col_0 == df.shape[0]:
delete_col.append(col)
delete_col # 24 号和 108 号英雄从来没有被选择过

#删去这两个英雄所在的列
df.drop(delete_col, axis=1, inplace=True)

#逻辑回归
# 调整前模型
l_model = LogisticRegression()
l_model.fit(train_x, train_y)
predict = l_model.predict(train_x)
fpr, tpr, thresholds = roc_curve(train_y, predict, pos_label=1)
auc(fpr, tpr)

# 调整后的逻辑回归模型
adjusted_l_model = LogisticRegression(solver='sag', C=0.1)
adjusted_l_model.fit(train_x, train_y)
predict = adjusted_l_model.predict(train_x)
fpr, tpr, thresholds = roc_curve(train_y, predict, pos_label=1)
auc(fpr, tpr)#逻辑回归的效果一般

#全连接神经网络
#导入建模所需库
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve

#所有数据+1之后，所有数据的-1都变为0，1变为2，方便建立模型
df = df + 1
train_x, train_y = df.loc['train'].iloc[:, :-2], label # 模型拆分
test_x = df.loc['test'].iloc[:, :-2

# 对数据标准化
std = StandardScaler()
std.fit(train_x)
train_x = std.transform(train_x)
test_x = std.transform(test_x)

train_x.shape, train_y.shape, test_x.shape

#调整参数前
model = Sequential(
[
layers.Dense(16, activation='relu'),
layers.Dense(2, activation='sigmoid')
]
)
 model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) # 配置损失函数和优化器
 model.fit(train_x, tf.one_hot(train_y,depth=2), batch_size=16, epochs=10, validation_split=0.2, verbose=0) # 训练模型

#计算auc值
y1_lr1 = model.predict(train_x)[:, 1]
fpr_lr1, tpr_lr1, thresholds_lr1 = roc_curve(train_y, y1_lr1)
roc_auc_lr1 = auc(fpr_lr1, tpr_lr1)
roc_auc_lr1

# 调整参数后的模型
# 主要调整了网络结构，batchsize和epochs
adjusted_model = Sequential(
[
layers.Dense(32, activation='relu'),
layers.Dense(64, activation='relu'),
layers.Dense(2, activation='sigmoid')
]
)
adjusted_model.compile(loss='binary_crossentropy', metrics=['accuracy']) # 配置损失函数和优化器
adjusted_model.fit(train_x, tf.one_hot(train_y,depth=2), epochs=40, validation_split=0.2, verbose=0) # 训练模型

#计算auc值
y1_lr2 = adjusted_model.predict(train_x)[:, 1]
fpr_lr2, tpr_lr2, thresholds_lr2 = roc_curve(train_y, y1_lr2)
roc_auc_lr2 = auc(fpr_lr2, tpr_lr2)
roc_auc_lr2

#绘制ROC曲线
plt.plot(fpr_lr1, tpr_lr1, fpr_lr2, tpr_lr2, lw=2, alpha=.6)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.legend(["ANN_1 (AUC {:.4f})".format(roc_auc_lr1),
"ANN_2 (AUC{:.4f})".format(roc_auc_lr2)], fontsize=8, loc=2)
