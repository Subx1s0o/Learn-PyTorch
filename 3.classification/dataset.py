from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)


# plt.scatter(x=X[:, 0],  
#             y=X[:, 1],  
#             c=y,       
#             cmap=plt.cm.RdYlBu)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Scatter plot of make_circles dataset")
# plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42 )