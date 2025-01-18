import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 设置字体
rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 1. 生成一个简单的二分类数据集
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 2. 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建 LDA 模型并训练
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 4. 在测试集上进行预测
y_pred = lda.predict(X_test)

# 5. 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy * 100:.2f}%")

# 6. 可视化决策边界和数据点
def plot_decision_boundary(X, y, model):
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和数据点
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('LDA 决策边界')
    plt.show()

# 可视化训练集的决策边界和数据点
plot_decision_boundary(X_train, y_train, lda)

# 7. 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()