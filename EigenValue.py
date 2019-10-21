#(1023 homework)針對數據集畫直方圖&折線圖 * 2(有套sklearn畫圖&沒套套件畫圖)

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn import preprocessing,tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


avocado=pd.read_csv("avocado.csv")

x=pd.DataFrame([avocado["Total Volume"],
                avocado["Total Bags"],
                avocado["AveragePrice"],
                avocado["Small Bags"],
                avocado["Large Bags"],
                avocado["XLarge Bags"],]).T


y=avocado["type"]

#切割成75%訓練集，25%測試集
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#standardizing the data
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

#Eigendecomposition of the convariance matrix(eigenvalues->lamda)
cov_mat=np.cov(X_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print('\nEigenvalues :\n%s' % eigen_vals)

#total and explained variance(直方圖)
tot = sum(np.abs(eigen_vals))
var_exp = [(i / tot) for i in sorted(np.abs(eigen_vals), reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, eigen_vals.size + 1), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, eigen_vals.size + 1), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Feature transformation(將資料轉換到新座標軸,特徵做投影)
#Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(reverse=True)

W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Peojection matrix W:\n', W)

#visualize it using (pca_scatter)
# Z-normalize data
sc = StandardScaler()
Z = sc.fit_transform(x)
Z_pca = Z.dot(W)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y.values), colors, markers):
    plt.scatter(Z_pca[y.values==l, 0], 
                Z_pca[y.values==l, 1], 
                c=c, label=l, marker=m)

plt.title('Z_pca')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()






