import numpy as np
import matplotlib.pyplot as plt
# X is assumed to be dataframe(m,n)
# m,n is the usual notation
# Y is assumed to be dataframe(m,1)

def initialize(dim):
	w = np.zeros((dim,1))
	b = 0
	return w,b

def sigmoid(z):
	s = 1/(np.exp(-z)+1)
	return s

def propagate(w,b,X,Y):
	m = X.shape[0]
	Z = np.dot(X,w) + b
	A = sigmoid(Z)
	cost = np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))*(-1)/m
	dw = np.dot(X.T,A-Y)/m
	db = np.sum(A-Y)/m
	grad = {"dw": dw,"db": db}
	return grad,cost

def cost_forward(w, b, X, Y):
	m = X.shape[0]
	Z = np.dot(X,w) + b
	A = sigmoid(Z)
	cost = np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))*(-1)/m
	return cost

def optimize(w,b,Train,Test,num_iter,learning_rate):
	X=Train[0]
	Y=Train[1]
	X_test=Test[0]
	Y_test=Test[1]
	dw = np.zeros((X.shape[1],1))
	db = 0
	cost = 0
	train_hist=[]
	test_hist=[]
	for i in range(num_iter):
		grad, cost = propagate(w,b,X,Y)
		train_hist.append(cost)
		test_c=cost_forward(w, b, X_test, Y_test)
		test_hist.append(test_c)
		dw = grad["dw"]
		db = grad["db"]
		w = w - learning_rate*dw
		b = b - learning_rate*db
	params = {"w": w, "b": b}
	return params, train_hist, test_hist

def logreg_train(X, Y, num_iter=2000, learning_rate=0.01):
	m, n=X.shape
	X = X.to_numpy()
	Y = Y.to_numpy().reshape((X.shape[0],1))
	Dataset=np.hstack((X, Y))
	np.random.shuffle(Dataset)
	X, Y=Dataset[:, 0:n], Dataset[:, n:]
	train_end=int(X.shape[0]*0.8)
	X_train, Y_train= X[0:train_end, :], Y[0:train_end, :]
	X_test, Y_test= X[train_end:, :], Y[train_end:, :]
	Train=X_train, Y_train.reshape(X_train.shape[0], 1)
	Test=X_test, Y_test.reshape(X_test.shape[0], 1)
	w,b = initialize(X_train.shape[1])
	params, train_hist, test_hist = optimize(w,b,Train,Test,num_iter,learning_rate)
	plt.figure()
	plt.plot(train_hist, linestyle='solid', color='red', label='Training Cost')
	plt.plot(test_hist, linestyle='dashed', color='blue', label='Testing Cost')
	plt.legend()
	plt.savefig("mlgym1/static/assets/logreg.png")
	return params
	
def logreg_predict(X_test, params):
	X = X_test.to_numpy()
	m = X.shape[0]
	Y_predict = sigmoid(np.dot(X,params["w"])+params["b"])
	Y_predict = (Y_predict > 0.5).astype(int)
	return Y_predict

def accuracy(Y_predict,Y_test):
	Y_test = Y_test.to_numpy()
	Y_test = Y_test.reshape((Y_predict.shape[0],1))
	c = 0
	m = Y_test.shape[0]
	for i in range(Y_test.shape[0]):
		if Y_test[i,0]==Y_predict[i,0]:
			c+=1
	return c/m