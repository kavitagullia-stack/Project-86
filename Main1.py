import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
x,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.60)
clf=SGDClassifier(loss="hinge",alpha=0.01,max_iter=200)
clf.fit(x,y)
x1=np.linspace(-1,5,10)
y1=np.linspace(-1,5,10)
x2,x3=np.meshgrid(x1,y1)
z=np.empty(x2.shape)
for (i,j),value in np.ndenumerate(x2):
    X2=value
    X3=x3[i,j]
    p=clf.decision_function([[X2,X3]])
    z[i,j]=p[0]
levels=[-1.0,0.0,1.0]
linestyle=["dashed","solid","dashed"]
colors="k"
plt.contour(x2,x3,z,levels,colors=colors,linestyles=linestyle)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired,edgecolor="black",s=20)
plt.axis("tight")
plt.show()


