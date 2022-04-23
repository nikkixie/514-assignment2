Code for all classifiers are contained in main.py

The original dataset is sliced into three binary classification problems: 'H' and 'K', 'M' and 'Y', 'B' and 'I'. We will use
five different classifiers, KNN, Decision Tree, Random Forest, SVM and ANN to predict the correct label.

To configure which sub-problem dataset to run, tweak the input at line 472 - 474.

All models are in train_models() func. See the comment above each classifier to see which model it is. By default, the problem currently
run on all sub-problem dataset with dimension reduction. To tweak dimension reduction on the dataset, simply comment out the part
without dimension reduction and uncomment out the part with dimension reduction.

For ANN model specifically, I used autoencoder to reduce the feature dimension. Autoencoders are trained in a different file
autoencoder.py. The trained model is saved with their names specifying they are for which dataset. For example, set1_encoder.h5 is
the autoencoder for set 1, i.e the classification problem for H and K. To load the autoencododers, tweak the name at line 336 in main.py
