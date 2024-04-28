import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\IEECS-AIML\heart_disease_data.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_pred = y_pred.reshape(-1,1)

y_test = y_test.reshape(-1,1)

print(np.concatenate((y_pred,y_test),1))

from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
