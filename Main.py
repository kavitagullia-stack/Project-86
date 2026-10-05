import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
sb.set_style("dark")
mpl.style.use(['https://gist.githubusercontent.com/BrendanMartin/01e71bb9550774e2ccff3af7574c0020/raw/6fa9681c7d0232d34c9271de9be150e584e606fe/lds_default.mplstyle'])
mpl.rcParams.update({"figure.figsize": (8,6),"axes.titlepad":22.0})
print(f"Taget variables : {dataset["target_names"]}")
(unique,counts)=np.unique(dataset["target"],return_counts=True)
print(f"Unique values of target variable are {unique}")
print(f"counts of target variable are {counts}")
sb.barplot(x=dataset["target_names"],y=counts)
plt.title("Target variable count and dataset")
plt.show()