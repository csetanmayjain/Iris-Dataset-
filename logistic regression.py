import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
iris = datasets.load_iris()
x = iris.data[:,:]  
y = iris.target
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x, y)  # we create an instance of Neighbours Classifier and fit the data.
pred=logreg.predict(x)
correct,ac,bc,cc=0,0,0,0
for i in range(0,len(y)):
    if(pred[i]==y[i]):
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