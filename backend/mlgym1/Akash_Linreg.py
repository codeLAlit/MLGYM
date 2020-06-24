import numpy as np
import matplotlib.pyplot as plt
def cost_fn(x,y,theta,lamb=0):
    y=np.array(y)
    m=np.size(x,0)
    n=np.size(x,1)
    one=np.ones((m,1),dtype=int)
    x=np.concatenate((one,x),axis=1)
   
    theta_null=theta.copy()
    theta_null[0]=0
 
    inter=np.matmul(x,theta)
    inter=inter-y
    J=np.matmul(inter.transpose(),inter)/(2*m)+((lamb/(2*m))*np.dot(theta_null.transpose(),theta_null))
    J=J[0][0]
    return J

def grad_des(Train, Test,theta,lamb=0,alpha=0.00009,maxiter=150000,max_error=10**-4):
    x, y=Train[0], Train[1]
    x_test, y_test=Test[0], Test[1]
    y=np.array(y)
    m=np.size(x,0)
    n=np.size(x,1)
    x2=x.copy()
    # if(m<10000):
    #  one=np.ones((m,1),dtype=int)
    #  x=np.concatenate((one,x),axis=1)
    #  theta=np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),np.matmul(x.transpose(),y))
    #  return theta
     
    y=np.array(y)
    m=np.size(x,0)
    n=np.size(x,1)    
    x2=x.copy()    
    one=np.ones((m,1),dtype=int)
    x=np.concatenate((one,x),axis=1)
    theta_null=theta.copy()
    theta_null[0]=0        
    inter=np.matmul(x,theta)-y    
    inter2=np.matmul(x.transpose(),inter)
    i=0
    J2=0
    J1=cost_fn(x2,y,theta,lamb)
    Jt=cost_fn(x_test, y_test, theta, lamb)
    Difference=100
    train_hist=[]
    test_hist=[]
    train_hist.append(J1)
    while(True):
     inter=np.matmul(x,theta)-y  
     inter2=np.matmul(x.transpose(),inter)
     theta=theta-(alpha/m)*(inter2+lamb*(theta_null))
     J2=cost_fn(x2,y,theta,lamb)
     train_hist.append(J2)
     Jt=cost_fn(x_test, y_test, theta, lamb)
     test_hist.append(Jt)
     Difference=J1-J2     
     if((Difference<=max_error)|(i>maxiter)):
      break
     i=i+1 
     J1=J2
    return theta, train_hist, test_hist

def linearreg(x,y,lamb=0,alpha=0.01,maxiter=1000,max_error=10**-4):

    m=x.shape[0]
    Train=x[0:int(m*0.8), :], y[0:int(m*0.8)]
    Test=x[int(m*0.8):, :], y[int(m*0.8):]
    n=x.shape[1]
    ini_theta=np.random.rand(n+1).reshape(n+1, 1)
    theta_final, train_hist, test_hist=grad_des(Train, Test, ini_theta,lamb,alpha,maxiter,max_error)
    plt.figure()
    plt.plot(train_hist, linestyle='solid', color='red', label='Training Cost')
    plt.plot(test_hist, linestyle='dashed', color='blue', label='Testing Cost')
    plt.legend()
    plt.savefig("mlgym1/static/assets/linreg.png")
    return theta_final

def linreg_predict(x, theta):
    m=x.shape[0]
    x2=x.copy()    
    one=np.ones((m,1),dtype=int)
    x=np.concatenate((one,x),axis=1)
    return x@theta
