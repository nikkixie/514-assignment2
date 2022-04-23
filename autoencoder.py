import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model

def train_autoencoder(train_x, train_y, test_x, test_y):
    n_inputs = train_x.shape[1]

    le = LabelEncoder()
    le.fit(train_y)
    y_train_enc = le.transform(train_y)
    y_test_enc = le.transform(test_y)

    visible = Input(shape=(n_inputs,))
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    n_bottleneck = round(float(n_inputs) / 4.0)
    bottleneck = Dense(n_bottleneck)(e)

    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs * 2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(train_x, y_train_enc, epochs=200, batch_size=16, verbose=2,
                        validation_data=(test_x, y_test_enc))

    encoder = Model(inputs=visible, outputs=bottleneck)  # compress feature dim
    encoder.save('set3_encoder.h5')





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

# train_autoencoder(train1_x, train1_y, test1_x, test1_y)
# train_autoencoder(train2_x, train2_y, test2_x, test2_y)
train_autoencoder(train3_x, train3_y, test3_x, test3_y)