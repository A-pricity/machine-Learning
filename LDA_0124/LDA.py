import numpy as np
from sklearn import datasets

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LDA:
    def __init__(self):
        self.S = None # covariate matrix 这个·S·就是协变量矩阵
        self.mu = {} # the mean of each class 每个类的均值
        self.prior = {} # the prior density of each class 先验概率密度
        self.labels = None # labels of data 样本标签
        self.sigma = {} # the covariate of each class 协变量

    def fit(self, X, y):
        # 获取所有的类别 get all the classes
        X = np.asarray(X)
        y = np.asarray(y)
        labels = np.unique(y)
        self.labels = labels
        n_samples = X.shape[0]
        print(labels)
        means = []
        for label in labels:
            # 计算每一个类别的均值
            tmp = np.mean(X[y == label], axis=0)
            means.append(tmp)
            self.mu[label] = tmp
            self.prior[label] = len(X[y == label])/n_samples
        if len(labels) == 2:
            mu = (means[0] - means[1])
            mu = mu[:,None] # 转换成列向量
            B = mu @ mu.T # 协方差矩阵
        else:
            total_mu = np.mean(X, axis=0)
            B = np.zeros((X.shape[1], X.shape[1]))
            for i, m in enumerate(means):
                n = X[y==i].shape[0]
                mu_i = m - total_mu
                mu_i = mu_i[:,None] # 转换成列向量
                B += n * np.dot(mu_i , mu_i.T)
        s_t = []
        for label,m in enumerate(means):
            s_i = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == label]:
                t = row - m
                t = t[:,None] # 转换成列向量
                s_i += t @ t.T
            s_t.append(s_i)
        S = np.zeros((X.shape[1], X.shape[1]))

        for s in s_t:
            S += s
        self.S = S

        # S^-1B进行特征分解
        S_inv = np.linalg.inv(S)
        S_inv_B = S_inv @ B
        eig_vals, eig_vecs = np.linalg.eig(S_inv_B)

        # 按照特征值大小排序
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        return eig_vecs

    def predict(self, x):
        x = np.asarray(x)
        decisions = {}
        inv = np.linalg.inv(self.S)
        delta = []

        if x.ndim == 1: # 单个样本 X is a single sample
            for l in self.lablels1:
                t = x @ inv @ self.mu[l] -0.5 * self.mu[l] @ inv @ self.mu[l]
                decisions[l] = t
                delta.append(t)
            inds_max = np.argmax(delta)
            return self.labels[inds_max]
        else: # 多个样本 X is a batch of samples
            ret = []
            for sam in x:
                for l in self.labels:
                    t = sam @ inv @ self.mu[l] -0.5 * self.mu[l] @ inv @ self.mu[l] + np.log(self.prior[l])
                    delta.append(t)
                inds_max = np.argmax(delta)
                ret.append(self.labels[inds_max])
                delta = []
            return ret

    def get_accuracy(self, X, y):
            return np.sum(self.predict(X) == y)/len(y)

# 构造数据集
def create_data(centers = 3, cluster_std=[1.0, 3.0, 2.5], n_samples = 150, n_features = 2):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, n_features=n_features)
    return X, y

if __name__ == '__main__':
    X, y = create_data(2, [1.0, 1.0],) # 2类2个特征
    # X, y = create_data(2, [1.0, 3.0],) # 2类2个特征
    # X, y = create_data([[2.0, 1.0], [15.0, 5.0], [31.0, 12.0]], [1.0, 3.0, 2.5]) # 3类2个特征
    np.random.seed(12)
    np.random.shuffle(X)
    np.random.seed(12)
    np.random.shuffle(y)

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    lda = LDA()
    eig_vecs = lda.fit(X_train, y_train)
    w = eig_vecs[:, :1] # 取前两个特征向量作为投影方向

    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(figsize=(10, 8))
    for point, pred in zip(X, y):
        # 画出原始数据散点图
        ax.scatter(point[0], point[1], c=colors[pred], alpha=0.5)
        # 画出数据点在w方向投影的图
        proj = (np.dot(point, w) * w) /np.dot(w.T, w)

        # 画出所有数据的投影点
        ax.scatter(proj[0], proj[1], c=colors[pred], alpha=0.5)
    plt.show()

    # predict
    print(lda.get_accuracy(X_test, y_test))