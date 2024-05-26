import numpy as np
from sklearn.linear_model import LinearRegression
model=LinearRegression()

x=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,3.5,4.5,5.5,6.5])
model.fit(x,y)
new_x=np.array([[6],[7],[8]])

prediction=model.predict(new_x)

for i,pred in enumerate(prediction):
    print(f" prediction for x={new_x[i][0]}:{pred}")
