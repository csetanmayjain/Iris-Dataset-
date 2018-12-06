import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets    #load iris datassset
import numpy as np
iris=datasets.load_iris()       
x=iris.data[:,:]
y=iris.target
np.save('new_iris',x)
a=np.load('new_iris.npy')
lr_model=svm.SVC()
lr_model.fit(x,y)
pred=lr_model.predict(x)
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