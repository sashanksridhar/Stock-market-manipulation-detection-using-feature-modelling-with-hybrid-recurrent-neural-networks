from keras.models import load_model
import csv
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
import pandas as pd


def calc_precision_recall(y_true, y_pred):
    # Convert predictions to series with index matching y_true
    # y_pred = pd.Series(y_pred, index=y_true.index)

    # Instantiate counters
    TP = 0
    FP = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(0,len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    # Calculate true positive rate and false positive rate
    # Use try-except statements to avoid problem of dividing by 0
    try:
        precision = TP / (TP + FP)
    except:
        precision = 1

    try:
        recall = TP / (TP + FN)
    except:
        recall = 1

    return precision, recall

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

Xtest = Xtest[:991]

Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[2], 4))

stacked_model = load_model("finalensemble.h5")

Xt = Xt[:len(Xt)-5]

Yt = Yt[:len(Yt)-5]

testX = np.asarray(Xt)

def predict_stacked_model(model, inputX,inpReg):
    X = [inputX for _ in range(len(model.input)-1)]
    X.append(inpReg)
    return model.predict(X)


probability_thresholds = np.linspace(0, 1, num=100)


yhat = predict_stacked_model(stacked_model, testX,Xtest)
lr_probs = []
# p = []


for x in Yt:
    if x ==1:
        lr_probs.append(x)

l2_precision_scores = []
l2_recall_scores = []
for j in probability_thresholds:
    p = []

    for x in np.nditer(yhat):
        xi = []

        if x < float(j):
            p.append(0)
        else:
            # lr_probs.append(x)
            p.append(1)

    # lr_precision, lr_recall, _ = precision_recall_curve(Yt, p)
    lr_precision, lr_recall= calc_precision_recall(Yt, p)
    # lr_f1, lr_auc = f1_score(Yt, p), auc(lr_recall, lr_precision)

    # l2_precision_scores.append(lr_precision[1])
    # l2_recall_scores.append(lr_recall[1])
    l2_precision_scores.append(lr_precision)
    l2_recall_scores.append(lr_recall)

# lr_probs = predict_stacked_model(stacked_model, testX,Xtest)
# keep probabilities for the positive outcome only

# predict class values

print(l2_recall_scores)
print(l2_precision_scores)
# print(Yt)

stacked_model = load_model("finalbiensemble.h5")
yhat1 = predict_stacked_model(stacked_model, testX,Xtest)

l2_precision_scores1 = []
l2_recall_scores1 = []
for j in probability_thresholds:
    p = []

    for x in np.nditer(yhat1):
        xi = []

        if x < float(j):
            p.append(0)
        else:
            # lr_probs.append(x)
            p.append(1)

    # lr_precision, lr_recall, _ = precision_recall_curve(Yt, p)
    lr_precision, lr_recall= calc_precision_recall(Yt, p)
    # lr_f1, lr_auc = f1_score(Yt, p), auc(lr_recall, lr_precision)

    # l2_precision_scores.append(lr_precision[1])
    # l2_recall_scores.append(lr_recall[1])
    l2_precision_scores1.append(lr_precision)
    l2_recall_scores1.append(lr_recall)


fig, ax = pyplot.subplots(figsize=(6,6))
no_skill = len(lr_probs) / len(Yt)
# print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline')

ax.plot(l2_recall_scores, l2_precision_scores, label='Hybrid ANN-LSTM')
ax.plot(l2_recall_scores1, l2_precision_scores1, label='Hybrid ANN-BiLSTM')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
# show the legend
ax.legend()
# show the plot
pyplot.show()

