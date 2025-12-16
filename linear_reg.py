import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    mu=np.mean(x)
    sigma=np.std(x)
    if sigma == 0:
        sigma = 1.0
    x_norm=(x-mu)/sigma
    return x_norm,mu,sigma

def predict(x,w,b):
    return w*x+b

def compute_cost(x,y,w,b):
    m=len(x)
    total_cost=0.0
    for i in range(m):
        error=predict(x[i],w,b)-y[i]
        total_cost+=error**2
    return total_cost/(2*m)

def compute_gradient(x,y,w,b):
    m=len(x)
    dj_dw=0.0
    dj_db=0.0
    for i in range(m):
        error=predict(x[i],w,b)-y[i]
        dj_dw+=error*x[i]
        dj_db+=error
    return dj_dw, dj_db

def gradient_descent(x,y,w,b,alpha,iterations):
    j_history=[]
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        cost=compute_cost(x,y,w,b)
        j_history.append(cost)
        if i%100==0:
            print(f"Iter: {i:4d}, Cost:{cost:.4f} , w:{w:.2f} ,b:{b:.2f}")
    return w,b,j_history

x_raw=np.array([500,800,1000,1200,1500],dtype=float)
y_train=np.array([90,100,350,500,700],dtype=float)

x_train,mu,sigma=normalize(x_raw)

w_init=0
b_init=0
alpha=0.01
iterations=1000

w,b,j_history=gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations)
print(f"\nFinal Parameters:")
print(f"w: {w:.4f}, b:{b:.4f}")

plt.figure()
plt.scatter(x_train,y_train)
x_line=np.linspace(min(x_train),max(x_train),100)
y_line=predict(x_line,w,b)
plt.plot(x_line,y_line)
plt.xlabel("Normalized house size(in sqft)")
plt.ylabel("Price(in thousand dollars)")
plt.show()

plt.figure()
plt.plot(j_history)
plt.xlabel("Iterations ")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

size=float(input("Enter House size: "))
size_norm,mu,sigma=normalize(x_raw)  
size_norm=(size-mu)/sigma  
price=predict(size_norm,w,b)
print(f"Predicted price : {price:.2f} thousand dollars")