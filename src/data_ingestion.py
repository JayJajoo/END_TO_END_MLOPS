import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

random_state = 42
test_size = 0.2

X,Y = load_iris(return_X_y=True,as_frame=True)
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,random_state=random_state)


PATH = os.path.abspath(__file__)

DATA_DIR = os.path.join(PATH,"..","data")

print(DATA_DIR)

train_set = pd.concat([X_train,y_train],axis = 1)

test_set = pd.concat([X_test,y_test],axis = 1)

train_set.to_csv(os.path.join(DATA_DIR,"train.csv"),index=False)
test_set.to_csv(os.path.join(DATA_DIR,"test.csv"),index=False)

print("Data Ingestion Completed!")