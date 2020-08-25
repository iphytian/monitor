# 特征数目选择
# The number of feature selection by compare SVM, RF, KNN

import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

path = './dataset_cnn/test0824.txt'
f = open(path,'r')

#划分数据与标签
x=[]
y=[]
a=[]
contents = f.readlines()
for i in range(1,11):
    for content in contents:
        value = content.split()
        x.append(value[1:i])
        y.append(value[11])

    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x,y,random_state=11,train_size=0.6, test_size=0.4)

    # 训练RF
    # train RF
    n_features = 9
    classifier = RandomForestClassifier(n_estimators=50, class_weight='balanced', max_features=None, max_depth=30,
                                        min_samples_split=2, bootstrap=True)
    classifier.fit(train_data, train_label)
    a1 = classifier.score(test_data, test_label)
    print(a1)

    # 训练KNN
    # train KNN
    classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    metric_params=None, n_jobs=None, n_neighbors=16, p=2, weights='distance')
    classifier.fit(train_data, train_label)
    a2 = classifier.score(test_data, test_label)
    print(a2)

    # 训练SVM
    # train SVM
    classifier = svm.SVC(C=4, kernel='rbf', gamma=0.005, decision_function_shape='ovr')
    classifier.fit(train_data, train_label)
    a3 = classifier.score(test_data, test_label)
    print(a3)

    a.append([a1,a2,a3])
    print(i)
    x=[]
    y=[]

output = open('C:\\Users\\hutia\\Desktop\\selected_feature\\test0724.txt','w')
for i in range(len(a)):
    for j in range(len(a[i])):
        output.write(str(a[i][j]))
        output.write('\t')
    output.write('\n')
output.close()

print(a)
'''