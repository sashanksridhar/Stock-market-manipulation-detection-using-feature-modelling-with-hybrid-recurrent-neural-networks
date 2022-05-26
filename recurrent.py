import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import classification_report, confusion_matrix
Xtrain = []
Ytrain = []
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
        # Xtrain.append(row[:len(row)-1])
        # Ytrain.append(row[len(row)-1:][0])
        Xtrain.append(row)

Xtest = []
Ytest = []
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
        # Xtest.append(row[:len(row)-1])
        # Ytest.append(row[len(row)-1:][0])
        Xtest.append(row)

seq_len = 2
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

print(Xtrain.shape)

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[2], 2))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[2], 2))

print(Xtrain.shape)

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (Xtrain.shape[1], 2)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True)) #this
regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 50, return_sequences=True))
# regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(1, activation='sigmoid'))

regressor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
regressor.fit(Xtrain,Ytrain,epochs=5)
# regressor.save("LSTM5.h5")

print(regressor.evaluate(Xtest,Ytest))

yhat = regressor.predict(Xtest)
p = []

for x in np.nditer(yhat):
    xi = []


    if x < float(0.5):
        p.append(0)
    else:
        p.append(1)



count = 0
for i in range(0,len(Ytest)):

    if Ytest[i]==p[i]:
        count +=1

print(count)
print((count/len(Ytest))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(Ytest, p))
print('\n')
print("=== Classification Report ===")
print(classification_report(Ytest, p))
print('\n')