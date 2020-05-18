import numpy as np
import pandas as pd
def linsvm_train(X_train,Y_train,iters,L_rate):
        X_train=np.array(X_train)
        m,n=X_train.shape
        Y_train=np.array(Y_train)
        Y_train.reshape(m,1)
        Y_train[Y_train==0]=-1; # svm requires the y's to be 1 or -1
        theta=np.zeros((n,1))
        # cost = 1000
        reg=1/iters
        while(iters>0):
            dw=np.zeros((n,1))
            for i in range(m):
                margin=1-Y_train[i]*(X_train[i].dot(theta))
                if(max(margin,0)==0):
                    dw+=reg*theta
                else:
                    dw+=reg*theta-Y_train[i]*(X_train[i][np.newaxis].T)
            theta=theta-L_rate*dw/m
            iters-=1
            # cost=0
            # for i in range(m):
            #     cost+=max(0,1-y[i][0]*(X_train[i].dot(theta)))
            # cost+=(theta.T).dot(theta)/2
            # cost/=m;

        return theta






def linsvm_predict(X_test,theta,bias=1):
    X_test=np.array(X_test)
    Y_test=X_test.dot(theta)
    if(bias==1):
        Y_test[Y_test>=0]=1
        Y_test[Y_test<0]=0
    else:
        Y_test[Y_test>0]=1
        Y_test[Y_test<=0]=0
    return (Y_test)