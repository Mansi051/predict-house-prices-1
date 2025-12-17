import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    return (X-mu)/sigma,mu,sigma

def predict(X,w,b):
    return X@w+b

def compute_cost(X,y,w,b):
    m=X.shape[0]
    errors=predict(X,w,b)-y
    return np.sum(errors**2)/(2*m)

def compute_gradient(X,y,w,b):
    m=X.shape[0]
    errors=predict(X,w,b)-y
    dj_dw=(X.T@errors)/m
    dj_db=np.sum(errors)/m
    return dj_dw,dj_db

def gradient_descent(X,y,w,b,alpha,iterations):
    J=[]
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(X,y,w,b)
        w-=alpha*dj_dw
        b-=alpha*dj_db
        J.append(compute_cost(X,y,w,b))
        if i%100==0:
            print(f"Iter {i:4d}: Cost {J[-1]:.4f}")
    return w,b,J

X_raw=np.array([[800,2],[1000,3],[1200,3],[1500,4],[1800,4]],dtype=float)
y=np.array([180,220,260,310,360],dtype=float)

X,mu,sigma=normalize(X_raw)
w=np.zeros(X.shape[1])
b=0.0
alpha=0.01
iterations=1000
w,b,J=gradient_descent(X,y,w,b,alpha,iterations)

print("\nFinal Parameters:")
print("w:  ",w)
print("b:",b)

x_new=np.array([[1200,3]],dtype=float)
x_new=(x_new-mu)/sigma
price=predict(x_new,w,b)

print(f"\nPredicted Price: {price[0]:.2f}")

plt.figure()
plt.plot(J)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()
