from sklearn import neighbors
from sklearn import datasets    #load iris datassset
import numpy as np
import matplotlib.pyplot as plt
iris=datasets.load_iris()       


x=iris.data[:,:]
np.save('new_iris',x)
a=np.load('new_iris.npy')
y=iris.target
knn=neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
pred=knn.predict(x)
correct,ac,bc,cc=0,0,0,0
for i in range(0,len(y)):
    if(pred[i] == y[i]):
        correct+=1    
print "Accuracy:- ",(float(correct)/len(y))*100
for i in range (0,150):
    if i<50 and pred[i]==y[i]:
        ac=ac+1
    elif i<100 and pred[i]==y[i]:
        bc=bc+1
    elif i>=100 and pred[i]==y[i]:
        cc=cc+1
print"Accuracy for Class 1:- ",(float(ac)/50)*100
print"Accuracy for Class 2:- ",(float(bc)/50)*100
print"Accuracy for Class 3:- ",(float(cc)/50)*100
p,q= x[:,0],x[:,1]
plt.scatter(p,q,c=y)
plt.plot(p,q)