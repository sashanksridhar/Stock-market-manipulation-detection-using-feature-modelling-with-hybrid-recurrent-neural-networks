import csv
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense, Input, Concatenate
from keras.models import load_model
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
Xtrain = []
Ytrain = []
X = []
Y = []

def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = 'E:\\desktop\\fyp\\ensemble_neural\\models_5\\' + str(i + 1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def define_stacked_model(members,regressor):

    for i in range(len(members)):

        model = members[i]
        for layer in model.layers:

            # layer.trainable = False

            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name


    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    ensemble_visible.append(regressor.input)
    print(len(ensemble_visible))
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    ensemble_outputs.append(regressor.output)
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_stacked_model(model, inputX, inpReg, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input)-1)]
    X.append(inpReg)
    # model.fit(X, inputy, epochs=300)
    history = model.fit(X, inputy,validation_split=0.33, epochs=5)
    model.save("finalensemble.h5")
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    plt.show()

def predict_stacked_model(model, inputX,inpReg):
    X = [inputX for _ in range(len(model.input)-1)]
    X.append(inpReg)
    return model.predict(X)

# generate 2d classification dataset
with open("normalized1.csv", 'r') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c==0:
            c+=1
            continue

        for j in range(0,len(row)):
            row[j] = float(row[j])
        X.append(row[:len(row)-1])
        Y.append(row[len(row)-1:][0])
        Xtrain.append(row)

Xtest = []
Ytest = []
Xt = []
Yt = []
# generate 2d classification dataset
with open("normalized2.csv", 'r') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c==0:
            c+=1
            continue

        for j in range(0,len(row)):
            row[j] = float(row[j])
        Xt.append(row[:len(row)-1])
        Yt.append(row[len(row)-1:][0])
        Xtest.append(row)

seq_len = 4
sequence_length = seq_len + 1

result = []
for index in range(len(Xtrain) - sequence_length):
    result.append(Xtrain[index: index + sequence_length])

result = np.array(result)


Xtrain = result[:, :-1]
Ytrain = []
for i in result:
    # print(i)
    Ytrain.append(i[0][len(i[0])-1])
# Ytrain = result[:, -1]
# print(Ytrain.shape)
print(Ytrain[0])
result1 = []
for index in range(len(Xtest) - sequence_length):
    result1.append(Xtest[index: index + sequence_length])
result1 = np.array(result1)


Xtest = result1[:, :-1]
Ytest = []
for i in result1:
    # print(i)
    Ytest.append(i[0][len(i[0])-1])

Ytest = Ytest[:991]

print(Xtrain.shape)
Xtrain = Xtrain[:3931]
Xtest = Xtest[:991]

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[2], 4))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[2], 4))

print(Xtrain.shape)

regInp = Input(shape=(Xtrain.shape[1], 4))
x = LSTM(100,return_sequences=True)(regInp)
x = Dropout(0.2)(x)
x = LSTM(units = 50, return_sequences = True)(x)
x = Dropout(0.2)(x)
x = LSTM(units = 50, return_sequences = True)(x)
x = Dropout(0.2)(x)
x = LSTM(units = 50, return_sequences = True)(x)
x = Dropout(0.2)(x)
x = LSTM(units = 50, return_sequences = True)(x)
x = Dropout(0.2)(x)
x = LSTM(units = 50, return_sequences = True)(x)
x = Dropout(0.2)(x)
x = LSTM(units = 50)(x)
x = Dropout(0.2)(x)
regOut = Dense(1,activation='sigmoid')(x)
regressor = Model(regInp,regOut)

n_members = 1
# members = load_all_models(n_members)
# print('Loaded %d models' % len(members))

# stacked_model = define_stacked_model(members,regressor)
stacked_model = load_model("finalensemble.h5")
X = X[:len(X)-5]
Xt = Xt[:len(Xt)-5]
Y = Y[:len(Y)-5]
Yt = Yt[:len(Yt)-5]
trainX = np.asarray(X)
testX = np.asarray(Xt)

# fit_stacked_model(stacked_model, trainX,Xtrain,Y)



print(stacked_model.evaluate([testX,Xtest],Ytest))

yhat = predict_stacked_model(stacked_model, testX,Xtest)
p = []

for x in np.nditer(yhat):
    xi = []


    if x < float(0.5):
        p.append(0)
    else:
        p.append(1)



count = 0
for i in range(0,len(Yt)):

    if Yt[i]==p[i]:
        count +=1

print(count)
print((count/len(Yt))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(Yt, p))
print('\n')
print("=== Classification Report ===")
print(classification_report(Yt, p))
print('\n')