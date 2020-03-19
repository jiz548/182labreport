#Author: Jiawen Zhou

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

target_2 = False

df = pd.read_csv(r'C:\Users\zhouj\Downloads\college-football-team-stats-2019\ReplicatedAcousticFeatures-ParkinsonDatabase.csv')
df1= df.drop(columns='ID')

if (target_2==True):
    df2 = df1.groupby(np.arange(len(df))//3).mean()
    df2 = df2.drop(columns='Recording')
    df1 = df2

m = df1.shape[0]
N = df1.shape[1]

#normalize data
df_mean = np.mean(df1, axis=1)
df_mean.resize=(df_mean.size,1)
temp = df1.sub(df_mean, axis='index')
NF = temp/np.sqrt(N)
#print(df1)
'''
gamma=1
d=np.zeros(m) #setting up the diagonal of D
for i in range(5):
    d[i] = gamma
C=np.eye(m)+np.diag(d)
W=1/N*np.dot(np.dot(np.transpose(NF),C),NF)
'''
#Wishart matrix, find eigenvlue to find outliers

W=1/N*(np.dot(np.transpose(NF),NF))
#C= 1/N*(np.dot(np.transpose(NF),NF))
eigvals, eigvecs = np.linalg.eig(W)
plt.hist(eigvals,density=True,bins=1000)

#plot density
rho = m/N
a = (1-np.sqrt(rho))**2
b = (1+np.sqrt(rho))**2
x = np.arange(a,b,0.001)
#sigma = 0.7
y = np.sqrt((x-a)*(b-x))/(2*np.pi*(x))
h = plt.plot(x, y, lw=2)
plt.show()

plt.hist(eigvals,density=True,bins=1000)
h = plt.plot(x, y, lw=2)
plt.axis([0,30,0,1])
plt.show()

if (target_2 == True):
    plt.hist(eigvals,density=True,bins=1000)
    h = plt.plot(x, y, lw=2)
    plt.axis([900,1000,0,0.05])
    plt.show()
else:
    plt.hist(eigvals,density=True,bins=1000)
    h = plt.plot(x, y, lw=2)
    plt.axis([2500,3000,0,0.05])
    plt.show()

df1 = df1.T
m = df1.shape[0]
N = df1.shape[1]

#normalize data
df_mean = np.mean(df1, axis=1)
df_mean.resize=(df_mean.size,1)
temp = df1.sub(df_mean, axis='index')
NF = temp/np.sqrt(N)

#one outlier found
u, s, vt = np.linalg.svd(NF, full_matrices=False)
mu= df_mean
Q= u[:,[0,1]]
beta_2d = np.dot(Q.T, temp)
beta_bar= np.dot(Q.T, mu)
beta_bar.shape=(beta_bar.size,1)
pk_2d=beta_2d+beta_bar
plt.figure(figsize=(9, 6))
plt.title("PC1 for Parkinson Data")
plt.xlabel("PC1")
target_x = pk_2d[0, :]
status_0 = []
status_1 = []
gender_0 = []
gender_1 = []
fix_y = []
if (target_2 == True):
    for i in range(len(df2['Status'])):
        if (df2['Status'][i] == 0):
            status_0.append(target_x[i])
        else:
            status_1.append(target_x[i])
        if (df2['Gender'][i] == 0):
            gender_0.append(target_x[i])
        else:
            gender_1.append(target_x[i])
    fix_y = [0]*80
    sns.scatterplot(pk_2d[0, :], fix_y, hue = df2['Status']);
    plt.show()
else:
    for i in range(len(df['Status'])):
        if (df['Status'][i] == 0):
            status_0.append(target_x[i])
        else:
            status_1.append(target_x[i])
        if (df['Gender'][i] == 0):
            gender_0.append(target_x[i])
        else:
            gender_1.append(target_x[i])
    fix_y = [0]*240
    sns.scatterplot(pk_2d[0, :], fix_y, hue = df['Status']);
    plt.show()

sns.scatterplot(pk_2d[0, :], fix_y, hue = df['Status']);
plt.hist([status_0, status_1], bins=30, stacked=True, normed = True)
plt.show()

sns.scatterplot(pk_2d[0, :], fix_y, hue = df['Gender']);
plt.hist([gender_0, gender_1], bins=30, stacked=True, normed = True)
plt.show()

#plt.hist([gender_0, gender_1], bins=30, stacked=True, normed = True)

