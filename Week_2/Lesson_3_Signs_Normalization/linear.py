import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train = pandas.read_csv(filepath_or_buffer='../perceptron-train.csv', header=None)
train_target = train[0]
train_attributes = train.ix[:, 1:]

test = pandas.read_csv(filepath_or_buffer='../perceptron-test.csv', header=None)
test_target = test[0]
test_attributes = test.ix[:, 1:]

clf = Perceptron(random_state=241)
clf.fit(X=train_attributes, y=train_target)
predictions = clf.predict(test_attributes)
accuracy = accuracy_score(test_target, predictions)

train_attributes_scaled = scaler.fit_transform(train_attributes)
test_attributes_scaled = scaler.fit_transform(test_attributes)

clf_scaled = Perceptron(random_state=241)
clf_scaled.fit(X=train_attributes_scaled, y=train_target)
predictions_scaled = clf_scaled.predict(test_attributes_scaled)
accuracy_scaled = accuracy_score(test_target, predictions_scaled)

print 'Accuracy: {}, Scaled accuracy: {}\nDifference: {}'.format(accuracy, accuracy_scaled, accuracy_scaled - accuracy)
