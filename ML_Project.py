import pandas as pd  # 데이터 불러오기
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/황준하/PycharmProjects/pythonProject/국가건강검진_혈압혈당데이터.csv', sep=',')
print(df)


res = df.loc[:,['SEX','DIS','SBP']]  # DIS가 2인(고혈압인) 사람들의 최고혈압 데이터 필터링
ft= res['DIS'].isin([2])
res2=res[ft]
print(res2)

res2['SBP'].describe()  # 고혈압인 사람들의 최고혈압 통계

ft2=res['DIS'].isin([4])   # 고혈압과 정상인의 최고혈압 비교
nomal=res[ft2]
boxplot = plt.figure()
axes= boxplot.add_subplot(1,1,1)
axes.boxplot([res2['SBP'],nomal['SBP']],labels=['Hypertension','Nomal'])
axes.set_xlabel('DIS')
axes.set_ylabel('SBP')
axes.set_title('Boxplot of SBP by DIS Status')




df.groupby('DIS')['SBP'].mean()  # 고혈압/당뇨병 진료여부에 따른 최고혈압 평균


df.groupby('DIS')['DBP'].mean()  # 고혈압/당뇨병 진료여부에 따른 최저혈압 평균

df.groupby(['BTH_G','DIS'])['SBP'].mean()  # 연령별 고혈압/당뇨병 진료여부에 따른 최고혈압 평균

df.groupby(['SEX','BTH_G','DIS'])['SBP'].mean() # 성별과 연령별 고혈압/당뇨병 진료여부에 따른 최고혈압 평균

df.groupby(['SEX','BTH_G','DIS'])['FBS'].mean() # 성별과 연령별 고혈압/당뇨병 진료여부에 따른 공복혈당 평균

Hy_ft = df['DIS'].isin([2])  # 연령별 고혈압진료를 받은 사람 수
Hy_res=df[Hy_ft]
Hy_res.groupby('BTH_G')['DIS'].count()

diab_ft = df['DIS'].isin([3])  # 연령별 당뇨병진료를 받은 사람 수
diab_res=df[diab_ft]
diab_res.groupby('BTH_G')['DIS'].count()

df.groupby(['SEX','DIS'])['BMI'].mean()  # 성별과 고혈압/당뇨병 진료 여부에 따른 BMI 평균

sc_plot=plt.figure(figsize=(20,10))   # 고혈압 환자의 BMI에 따른 최고 혈압
axes = sc_plot.add_subplot(1,1,1)
axes.scatter(x=Hy_res['BMI'],y=Hy_res['SBP'])

Hy_res2=Hy_res.iloc[0::60,:]

sc_plot=plt.figure(figsize=(10,10))   # 고혈압 환자의 BMI에 따른 최고 혈압
axes = sc_plot.add_subplot(1,1,1)
axes.scatter(x=Hy_res2['BMI'],y=Hy_res2['SBP'])

import seaborn as sns   # 각 BMI와 최고 혈압에서의 고혈압 환자 빈도
hexplot=sns.jointplot(x=Hy_res2['BMI'],y=Hy_res2['SBP'], data='DIS', kind="hex")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np

clf = MLPClassifier(solver='adam', max_iter=1000,  alpha=1e-5,
                     hidden_layer_sizes=(5, 5, 5, 5), random_state=1)

X = df.drop(['FBS','BMI','DIS'], axis=1)
Y = df['DIS']
clf.fit(X, Y)

test_X = np.array([[1,1,127,79]], dtype=np.int64)
Filter3= df['DIS'].isin([4])
test_y=df[Filter3]

predict_y = clf.predict(test_X)
print(predict_y)
print(classification_report([4], predict_y))