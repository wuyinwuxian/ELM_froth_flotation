import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ELM():
    # 输入数据集X、标签Y、隐含层神经元个数m、控制参数L
    def __init__(self, X, Y, m, L):
        self.X = X
        self.Y = Y
        self.m, self.L = m, L

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # 训练函数，随机w,b 计算H、beta
    def TRAIN_beta(self):
        n, d = self.X.shape
        self.w = np.random.rand(d, self.m)
        self.b = np.random.rand(1, self.m)
        H = self.sigmoid(np.dot(self.X, self.w) + self.b)
        # self.beta = np.dot(np.linalg.inv(np.identity(self.m) / self.L + np.dot(H.T, H)),
        #                    np.dot(H.T, self.Y))  # 加入正则化且 n >> m
        self.beta = np.dot(np.linalg.pinv(H), self.Y) # 不加入正则化
        print('TRAIN FINISH beta_shape ', self.beta.shape)

    # 测试函数，计算方法
    def TEST(self, x):
        H = self.sigmoid(np.dot(x, self.w) + self.b)  # 使用测试集计算H，其中w、b是之前随机得到的
        result = np.dot(H, self.beta)
        return result

def normalization(data):
    col = data.shape[1]
    ans = np.zeros_like(data)
    for i in range(col):
        _range = np.max(data[:,i]) - np.min(data[:,i])
        ans[:,i] = (data[:,i] - np.min(data[:,i])) / _range
    return ans

# 8分类
# classes = 8
# pd_X = pd.read_csv('Features38_8_X.csv',header=None).iloc[:,[2, 6, 9, 13, 17]]  # 色调，蓝色均值，相对红色分量，粗度，高频能量
# pd_Y = pd.read_csv('Features38_8_Y.csv',header=None)-1  # 下标从0开始

# 三分类
classes = 3
pd_X = pd.read_csv('FS_X.csv',header=None).iloc[:,[2, 6, 9, 13, 17]]  # 色调，蓝色均值，相对红色分量，粗度，高频能量
pd_Y = pd.read_csv('FS_Y.csv',header=None)-1  # 下标从0开始

X = pd_X.values
X = normalization(X)
Y = pd_Y.values.flatten()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

test_size = Y_test.size
train_size = Y_train.size

# 训练时将标签改为OneHot编码效果比较好
Y_onehot = np.eye(classes)[Y_train]

# 使用时请自己选择神经元个数，如果要使用正则化的请指定最后的控制参数
elm = ELM(X_train, Y_onehot, 80, 2.0)
elm.TRAIN_beta()

# 训练集的误差
predict1 = elm.TEST(X_train)
predict11 = np.argmax(predict1, axis = 1) # OneHot编码形式 取每行最大值的索引即类别

acc1 = np.sum(predict11 == Y_train)/train_size
print("训练集精度 :",acc1)

# 进行测试
predict0 = elm.TEST(X_test)
predict00 = np.argmax(predict0, axis = 1) # OneHot编码形式 取每行最大值的索引即类别

acc = np.sum(predict00 == Y_test)/test_size
print('测试集精度 :', acc)