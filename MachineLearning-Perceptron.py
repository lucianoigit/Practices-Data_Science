import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def __init__(self,eta=0.01,n_iter=50,random_state=1):
    self.eta=eta
    self.n_iter=n_iter
    self.random_state=random_state
    
def fit(self,X,y):
    rgen=np.random.RandomState(self.random_state)
    self.w_=rgen.normal(loc=0.0,scale=0.01,sie=1+X.shape[1])
    self.errors_=[]
    
    
    for _ in range(self.n_iter):
        errors=0
        for xi,target in zip(X,y):
            update=self.eta*(target-self.predict(xi))
            self.w_[1:]+=update*xi
            self.w_[0]+=update
            errors+=int(update!=0.0)
            
        self.errors_.append(errors)
    return self

def net_input(self,X):
    return np.dot(X,self.w_[1:])+self.w_[0]

def predict(self,X):
    return np.where(self.net_input(X)>=0.0,1,-1)


# Entreno un modelo de perceptron en el conjunto de datos iris

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()

# Selecciono setosa y versicolor

y=df.iloc[0:100,4].values
y=np.where(y=="Iris-setosa",-1,1)

# Extraigo los datos de longitud de sepalo y longitud de petalo

X=df.iloc[0:100,[0,2]].values

# Represento los datos

plt.scatter(X[:50,0],X[:50,1],color="red",marker="o",label="setosa")

plt.scatter(X[50:100,0],X[50:100,1],color="blue",marker="x",label="versicolor")
                   
plt.xlabel("sepal legth[cm]")
plt.ylabel("petal legth[cm]")
plt.legend(loc="upper left")

plt.show()    
    


