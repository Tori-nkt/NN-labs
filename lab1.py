import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd



from tensorflow import keras
# Reading data
data_train = pd.read_csv('train1.csv')
data_test = pd.read_csv('test.csv')


from sklearn.preprocessing import LabelEncoder
label_encoder_sex = LabelEncoder()

data_train.iloc[:,3]  = label_encoder_sex.fit_transform(data_train.iloc[:,3])
data_test.iloc[:,3] = label_encoder_sex.fit_transform(data_test.iloc[:,3])


columns = ['Sex','SibSp','Parch','Ticket','Survived']
data_train = data_train[['Age','Fare','Sex','SibSp','Parch','Ticketclass','Survived']]
data_test = data_test[['Age','Fare','Sex','SibSp','Parch','Ticketclass','Survived']]

X_test = data_test.iloc[:,0:6]
y_test = data_test.iloc[:,6]
X_train = data_train.iloc[:,0:6]
y_train = data_train.iloc[:,6]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



model = Sequential()  # создание модели
model.add(Dense(input_dim=X_train.shape[1], units=128,
                kernel_initializer='uniform', bias_initializer='zeros')) # добавление входного шара
model.add(Activation('relu'))

for i in range(0, 2):
    model.add(Dense(units=64, kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(.25))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))

#binary_crossentropy
#mse
#categorical_crossentropy
#hinge
Loss = 'hinge'
opt = keras.optimizers.Adam(0.001)

model.compile(optimizer=opt,
              loss=Loss,
              metrics=['accuracy'])

from keras.models import model_from_json
json_file = open('model_msesigmoid.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/weights_msesigmoid.h5")
print("Loaded model from disk")

# evaluate loaded model on test data

loaded_model.compile(loss=Loss, optimizer=opt, metrics=['accuracy'])
scores = loaded_model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))

#getting predictions of test data
prediction = loaded_model.predict(X_test).tolist()
print(prediction)





