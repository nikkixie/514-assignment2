import time

import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model



import statistics
from sklearn.metrics import accuracy_score


def train_models(train_x, train_y, test_x, test_y):
    from sklearn import svm
    from sklearn.svm import SVC
    from mlxtend.feature_selection import SequentialFeatureSelector as sfs
    from sklearn.ensemble import RandomForestRegressor

    #First, KNN MODEL without dim reduction
    # k_range = range(1, 31)
    # k_scores = []
    # best_param = 1
    # best_score = -1
    # start = time.time()
    #
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, train_x, train_y, cv=5, scoring='accuracy')
    #     k_scores.append(scores.mean())
    #     if (scores.mean() > best_score):
    #         best_score = scores.mean()
    #         best_param = k
    #
    # print("Best K found is " + repr(best_param))
    # #final validation on original data
    # knn = KNeighborsClassifier(n_neighbors=best_param)
    # knn.fit(train_x, train_y)
    # pred = knn.predict(test_x)
    # accuracy = accuracy_score(test_y, pred)
    # end = time.time()
    # print("KNN Final validation without dimension reduction: " + repr(accuracy))
    # print("time cost for KNN: " + repr(end - start))
    #
    # # plot to see clearly
    # plt.plot(k_range, k_scores)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy Without Dim Reudction')
    # plt.show()




    # #Dimension reduction, simple quality filtering, find the feature with least var
    start =time.time()
    red_range = range(1,13)
    dim_reduced_x = train_x
    dim_reduced_x_test = test_x

    for r in red_range:
        least_var_col = -1
        least_var = 10000
        for idx, data in dim_reduced_x_test.iteritems():
            curr_var = data.var()
            if curr_var < least_var:
                least_var = curr_var
                least_var_col = idx

        dim_reduced_x= dim_reduced_x.drop(columns = least_var_col)
        dim_reduced_x_test = dim_reduced_x_test.drop(columns = least_var_col)

    k_range = range(1, 31)
    k_scores = []
    best_param = 1
    best_score = -1

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, dim_reduced_x, train_y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
        if (scores.mean() > best_score):
            best_score = scores.mean()
            best_param = k

    # plot to see clearly

    #final validation on dim reduced data
    knn = KNeighborsClassifier(n_neighbors=best_param)
    knn.fit(dim_reduced_x, train_y)
    pred = knn.predict(dim_reduced_x_test)
    dimreduced_acc = accuracy_score(test_y, pred)
    end = time.time()

    print("KNN Final validation with dimension reduction: " + repr(dimreduced_acc))
    print("time cost for KNN: " + repr(end - start))

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy With Dim Reudction')
    plt.show()

    print("Best K found is " + repr(best_param))


    #
    # #Decision Tree 1)gini/entropy. 2)Depth 3) min_sample
    # Without dim reduction
    # start = time.time()
    # depth_range = range(1, 15)
    # depth_scores = []
    # best_param = 1
    # best_score = -1
    #
    # for d in depth_range:
    #     clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=d, min_samples_leaf=5)
    #     scores = cross_val_score(clf_gini, train_x, train_y, cv=5, scoring='accuracy')
    #     depth_scores.append(scores.mean())
    #     if (scores.mean() > best_score):
    #         best_score = scores.mean()
    #         best_param = d
    #
    # dt = DecisionTreeClassifier(criterion="gini", random_state=40, max_depth=best_param, min_samples_leaf=5)
    # dt.fit(train_x, train_y)
    # pred = dt.predict(test_x)
    # score = accuracy_score(test_y, pred)
    # end = time.time()
    #
    # print("Decision Tree final validation without dimension reduction. Best depth with gini is:" + repr(score))
    # print("time cost for DT: " + repr(end - start))
    #
    # # plot to see clearly
    # plt.plot(depth_range, depth_scores)
    # plt.xlabel('Value of Depth for Decision Tree')
    # plt.ylabel('Cross-Validated Accuracy Without Dim Reudction')
    # plt.show()



    #
    # #Dim reduction
    start = time.time()
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=4,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=5)

    # Perform SFFS
    sfs1 = sfs1.fit(train_x, train_y)
    saved_col = list(sfs1.k_feature_idx_)
    reduced_x = train_x.iloc[:, saved_col]
    reduced_test_x = test_x.iloc[:, saved_col]

    depth_range = range(1, 15)
    depth_scores = []
    best_param = 1
    best_score = -1

    for d in depth_range:
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=d, min_samples_leaf=5)
        scores = cross_val_score(clf_gini, reduced_x, train_y, cv=5, scoring='accuracy')
        depth_scores.append(scores.mean())
        if (scores.mean() > best_score):
            best_score = scores.mean()
            best_param = d

    dt = DecisionTreeClassifier(criterion="gini", random_state=40, max_depth=best_param, min_samples_leaf=5)
    dt.fit(reduced_x, train_y)
    pred = dt.predict(reduced_test_x)
    score = accuracy_score(test_y, pred)
    end = time.time()
    print("time cost for DT: " + repr(end - start))
    print("Decision Tree final validation with dimension reduction. Best depth with gini is: " + repr(score))

    # plot to see clearly
    plt.plot(depth_range, depth_scores)
    plt.xlabel('Value of Depth for Decision')
    plt.ylabel('Cross-Validated Accuracy With Dim Reudction')
    plt.show()

    print("Decision Tree final validation with dimension reduction. Best depth with gini is: " + repr(score))

    #third model, random forest
    # Without Dimension reduction
    # start = time.time()
    # depth_range = range(1, 15)
    # depth_scores = []
    # best_param = 1
    # best_score = -1
    #
    # for d in depth_range:
    #     clf_gini = RandomForestClassifier(criterion="gini", random_state=40, max_depth=d, min_samples_leaf=5)
    #     scores = cross_val_score(clf_gini, train_x, train_y, cv=5, scoring='accuracy')
    #     depth_scores.append(scores.mean())
    #     if (scores.mean() > best_score):
    #         best_score = scores.mean()
    #         best_param = d
    #
    # rf = RandomForestClassifier(criterion="gini", random_state=40, max_depth=best_param, min_samples_leaf=5)
    # rf.fit(train_x, train_y)
    # pred = rf.predict(test_x)
    # score = accuracy_score(test_y, pred)
    # end = time.time()
    # print("Random Forest Final validation without dim reduction. Best depth with gini is: " + repr(score))
    # print("time cost for RF: " + repr(end - start))
    #
    # # plot to see clearly
    # plt.plot(depth_range, depth_scores)
    # plt.xlabel('Value of Depth for Random Forest')
    # plt.ylabel('Cross-Validated Accuracy Without Dim Reudction')
    # plt.show()
    #


    #Dim reduction
    start = time.time()
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=4,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=5)

    # Perform SFFS
    sfs1 = sfs1.fit(train_x, train_y)
    saved_col = list(sfs1.k_feature_idx_)
    reduced_x = train_x.iloc[:, saved_col]
    reduced_test_x = test_x.iloc[:, saved_col]

    depth_range = range(1, 15)
    depth_scores = []
    best_param = 1
    best_score = -1

    for d in depth_range:
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=d, min_samples_leaf=5)
        scores = cross_val_score(clf_gini, reduced_x, train_y, cv=5, scoring='accuracy')
        depth_scores.append(scores.mean())
        if (scores.mean() > best_score):
            best_score = scores.mean()
            best_param = d

    rf = RandomForestClassifier(criterion="gini", random_state=40, max_depth=best_param, min_samples_leaf=5)
    rf.fit(reduced_x, train_y)
    pred = rf.predict(reduced_test_x)
    score = accuracy_score(test_y, pred)
    end = time.time()
    print("time cost for RF: " + repr(end - start))
    print("Random Forest final validation with dimension reduction. Best depth with gini is: " + repr(score))

    # plot to see clearly
    plt.plot(depth_range, depth_scores)
    plt.xlabel('Value of Depth for Random Forest')
    plt.ylabel('Cross-Validated Accuracy With Dim Reudction')
    plt.show()


    #
    # #fourth model, SVM
    # Without Dimension reduction
    # start = time.time()
    # kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']
    # kernel_scores = []
    # best_param = ''
    # best_score = -1
    #
    # for k in kernel_range:
    #     clf = svm.SVC(kernel=k)
    #     scores = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy')
    #     kernel_scores.append(scores.mean())
    #     if (scores.mean() > best_score):
    #         best_score = scores.mean()
    #         best_param = k
    # svm = svm.SVC(kernel = best_param)
    # svm.fit(train_x, train_y)
    # pred = svm.predict(test_x)
    # score = accuracy_score(test_y, pred)
    # end = time.time()
    #
    # print("SVM Final validation without dim reduction: " + repr(score) +" with " + best_param)
    # print("time cost for SVM: " + repr(end - start))
    #
    # plt.plot(kernel_range, kernel_scores)
    # plt.xlabel('Type of kernel for SVM')
    # plt.ylabel('Cross-Validated Accuracy Without Dim Reudction')
    # plt.show()



    # With dimension reduction
    # first we convert string labels to ints
    start = time.time()
    le = LabelEncoder()
    le.fit(train_y)
    y_train_enc = le.transform(train_y)

    #train a RFR to give us the gini importance
    model = RandomForestRegressor(random_state=42, max_depth=5)
    model.fit(train_x, y_train_enc)
    print(model.feature_importances_)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-4:]
    redu_trainx = train_x.iloc[:, indices]
    redu_testx = test_x.iloc[:, indices]

    kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_scores = []
    best_param = ''
    best_score = -1

    for k in kernel_range:
        clf = SVC(kernel=k)
        scores = cross_val_score(clf, redu_trainx, train_y, cv=5, scoring='accuracy')
        kernel_scores.append(scores.mean())
        if (scores.mean() > best_score):
            best_score = scores.mean()
            best_param = k

    svm = SVC(kernel=best_param)
    svm.fit(redu_trainx, train_y)
    pred = svm.predict(redu_testx)
    score = accuracy_score(test_y, pred)
    end = time.time()
    print("SVM Final validation with dim reduction: " + repr(score) + " with " + best_param)
    print("time cost for SVM: " + repr(end - start))

    plt.plot(kernel_range, kernel_scores)
    plt.xlabel('Type of kernel for SVM')
    plt.ylabel('Cross-Validated Accuracy With Dim Reudction')
    plt.show()


    # #fifth model, ANN
    # #Because ANN only accepts number input, we need to first convert the text labels to numbers using label encoder, the final
    # #result are in range[1,0]
    start = time.time()

    le = LabelEncoder()
    le.fit(train_y)
    y_train_enc = le.transform(train_y)
    y_test_enc = le.transform(test_y)
    #
    #Without Dim reduction
    # def create_clf(layers, neuron):
    #     #https://thinkingneuron.com/how-to-use-artificial-neural-networks-for-classification-in-python/
    #     #referred the above link to know how to use grid search to find optimal params
    #     classifier = Sequential()
    #     classifier.add(
    #         Dense(units=neuron, input_dim=train_x.shape[1], activation='relu'))
    #     #tweak number of hidden layers
    #     for idx in range(layers - 1):
    #         classifier.add(Dense(units=neuron/2,  activation='relu'))
    #
    #      #output and compilation
    #     classifier.add(Dense(units=1,  activation='sigmoid'))
    #     classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    #     return classifier
    #
    #
    # monitor = EarlyStopping(monitor='accuracy', min_delta=1e-3, patience=10, verbose=1, mode='auto',
    #                         restore_best_weights=True)
    #
    # layer_range = [2,3,4,5,6]
    # neuron_range = [4,6,8,10,12]
    #
    # scores = []
    # combinations = []
    # best_layer_count = -1;
    # best_neuron_count = -1;
    # best_score = -1;
    # #find best parameters(layer count and neuron count)
    # for layer in layer_range:
    #     for neuron in neuron_range:
    #
    #         classifierModel=KerasClassifier(build_fn = lambda:create_clf(layer, neuron),  batch_size = 20, epochs = 10 , callbacks=[monitor])
    #
    #         accuracy = cross_val_score(classifierModel, train_x, y_train_enc, cv = 5, scoring =
    #                                    "accuracy") #Without Dim reduction
    #         # accuracy = cross_val_score(classifierModel, X_train_encode, y_train_enc, cv = 5, scoring =
    #         #                            "accuracy")
    #         if(accuracy.mean()> best_score):
    #             best_layer_count = layer
    #             best_neuron_count = neuron
    #             best_score = accuracy.mean()
    #         scores.append(accuracy.mean())
    #         combinations.append(str(layer) + ", " + str(neuron))
    #
    # # Final validation on original data
    # clf = Sequential()
    # clf.add(Dense(units=best_layer_count, input_dim=train_x.shape[1], activation='relu'))
    # # tweak number of hidden layers
    # for idx in range(best_layer_count - 1):
    #     clf.add(Dense(units=best_neuron_count / 2, activation='relu'))
    # clf.add(Dense(units=1, activation='sigmoid'))
    # clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # clf.fit(train_x, y_train_enc,  batch_size=20 , epochs=10)
    # pred = clf.predict(test_x)
    # for idx in range(len(pred)):
    #     if pred[idx] >= 0.5:
    #         pred[idx] = 1
    #     else:
    #         pred[idx] = 0
    # score = accuracy_score(y_test_enc, pred)
    # print("ANN Final validation without dim reduction: " + repr(score) + " with layer count: " + repr(best_layer_count) + " and neuron count: " +
    #                                                                                                                       repr(best_neuron_count))
    # end = time.time()
    # print("used time without dim reduction" + repr(end - start))
    # #On filtered data
    #
    # plt.plot(combinations, scores)
    # plt.xlabel('Number of (layers, neurons) in ANN model')
    # plt.ylabel('Cross-Validated Accuracy Without Dim Reudction')
    # plt.show()



    # With Dimension reduction
    # before fitting the model, use pre-trained autoencoder to shrink the dimension size
    # train the autoencoders in autoencoder.py
    encoder = load_model('set1_encoder.h5') #change to corresponding encoder for each set
    #now we only keep four features through the bottleneck
    X_train_encode = encoder.predict(train_x)
    X_test_encode = encoder.predict(test_x)


    def create_clf(layers, neuron):
        #https://thinkingneuron.com/how-to-use-artificial-neural-networks-for-classification-in-python/
        #referred the above link to know how to use grid search to find optimal params
        classifier = Sequential()

        classifier.add(
            Dense(units=neuron, input_dim=X_train_encode.shape[1], activation='relu')) #With dim reduction data
        #tweak number of hidden layers
        for idx in range(layers - 1):
            classifier.add(Dense(units=neuron/2,  activation='relu'))

         #output and compilation
        classifier.add(Dense(units=1,  activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return classifier


    monitor = EarlyStopping(monitor='accuracy', min_delta=1e-3, patience=10, verbose=1, mode='auto',
                            restore_best_weights=True)

    layer_range = [2,3,4,5,6]
    neuron_range = [4,6,8,10,12]

    scores = []
    combinations = []
    best_layer_count = -1;
    best_neuron_count = -1;
    best_score = -1;
    start = time.time()
    #find best parameters(layer count and neuron count)
    for layer in layer_range:
        for neuron in neuron_range:

            classifierModel=KerasClassifier(build_fn = lambda:create_clf(layer, neuron),  batch_size = 20, epochs = 10 , callbacks=[monitor])
            accuracy = cross_val_score(classifierModel, X_train_encode, y_train_enc, cv = 5, scoring =
                                       "accuracy")
            if(accuracy.mean()> best_score):
                best_layer_count = layer
                best_neuron_count = neuron
                best_score = accuracy.mean()
            scores.append(accuracy.mean())
            combinations.append(str(layer) + ", " + str(neuron))


    #On filtered data


    #output and compilation
    clf = Sequential()
    clf.add(Dense(units=best_neuron_count, input_dim=X_train_encode.shape[1], activation='relu'))
    # tweak number of hidden layers
    for idx in range(best_layer_count - 1):
        clf.add(Dense(units=best_neuron_count / 2, activation='relu'))
    clf.add(Dense(units=1, activation='sigmoid'))
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    clf.fit(X_train_encode, y_train_enc,  batch_size=20 , epochs=10)
    pred = clf.predict(X_test_encode)
    for idx in range(len(pred)):
        if pred[idx] >= 0.5:
            pred[idx] = 1
        else:
            pred[idx] = 0
    score = accuracy_score(y_test_enc, pred)
    print("ANN Final validation with dim reduction: " + repr(score) + " with " + repr(best_layer_count) + " and neuron count: " +
                                                                                                                          repr(best_neuron_count))
    end = time.time()
    print("used time for ANN" + repr(end - start))

       #
    plt.plot(combinations, scores)
    plt.xlabel('Number of (layers, neurons) in ANN model')
    plt.ylabel('Cross-Validated Accuracy With Dim Reudction')
    plt.show()


# 	789 A	   766 B     736 C     805 D	 768 E	   775 F     773 G
#  	734 H	   755 I     747 J     739 K	 761 L	   792 M     783 N
#  	753 O	   803 P     783 Q     758 R	 748 S	   796 T     813 U
#  	764 V	   752 W     787 X     786 Y	 734 Z

#Data preprocessing HK(df1) - 1473, MY(df2) - 1578, BI(df3) - 1521
df = pd.read_csv("data/letter-recognition.data", header=None)
df1 = df.loc[(df[0] == 'H') | (df[0] == 'K')]
df2 = df.loc[(df[0] == 'M') | (df[0] == 'Y')]
df3 = df.loc[(df[0] == 'B') | (df[0] == 'I')]

train1, test1 = train_test_split(df1, test_size=0.1)
train2, test2 = train_test_split(df2, test_size=0.1)
train3, test3 = train_test_split(df3, test_size=0.1)


train1_x = train1.iloc[:,1:]
train1_y = train1.iloc[:,0]
test1_x = test1.iloc[:,1:]
test1_y = test1.iloc[:,0]

train2_x = train2.iloc[:,1:]
train2_y = train2.iloc[:,0]
test2_x = test2.iloc[:,1:]
test2_y = test2.iloc[:,0]

train3_x = train3.iloc[:,1:]
train3_y = train3.iloc[:,0]
test3_x = test3.iloc[:,1:]
test3_y = test3.iloc[:,0]

train_models(train1_x, train1_y, test1_x, test1_y)
train_models(train2_x, train2_y, test2_x, test2_y)
train_models(train3_x, train3_y, test3_x, test3_y)