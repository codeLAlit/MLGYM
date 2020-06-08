import numpy as np
import random
from scipy.special import expit



class parameters(object):
	def __init__(self,sizes):
		self.classes=sizes[-1]
		self.sizes=sizes
		self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		self.biases=[np.random.randn(y,1) for y in sizes[1:]]

params=parameters([10,5,2])

def gradient_descent(training_data,epochs,mini_batch_size,eta,lmbda,sizes):
	num_layers=len(sizes)
	global params
	params=parameters(sizes)
	training_data=list(training_data)
	n=len(training_data)
	
	for j in range(epochs):
		print("Training Epoch: {}".format(j+1))
		random.shuffle(training_data)
		mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
		for mini_batch in mini_batches:
			update(mini_batch,eta,lmbda)
		
	thetas=[]
	for w,b in zip(params.weights,params.biases):
		thetas.append(np.concatenate((b,w),axis=1))

	return thetas
	

def update(mini_batch,eta,lmbda):
	global params
	m=len(mini_batch)
	mini_batch=list(mini_batch)
	nabla_b=[np.zeros(b.shape) for b in params.biases]
	nabla_w=[np.zeros(w.shape) for w in params.weights]
	
	for x,y in mini_batch:
		#returns the change is all weights and biases
		dnb,dnw=backprop(x,y)
		nabla_b=[nb+dnb1 for nb,dnb1 in zip(nabla_b,dnb)]
		nabla_w=[nw+dnw1 for nw,dnw1 in zip(nabla_w,dnw)]
	params.biases=[b*(1-(lmbda*eta/m))-(eta/m)*nb for b,nb in zip(params.biases,nabla_b)]
	params.weights=[w*(1-(lmbda*eta/m))-(eta/m)*nw for w,nw in zip(params.weights,nabla_w)]

def backprop(x,y):
	global params
	nabla_b=[np.zeros(b.shape) for b in params.biases]
	nabla_w=[np.zeros(w.shape) for w in params.weights]
	activation=x.reshape((x.size,1))
	activations=[x.reshape((x.size,1))]
	
	for w,b in zip(params.weights,params.biases):
		z=np.dot(w,activation)+b
		activation=expit(z)
		activations.append(activation)
	
	#note that we have assumed y is not vectorized
	y=vectorize(y)
	
	#using cross entropy cost, therefore delta is not dependent on sigmoid_prime(z)
	delta=activations[-1]-y 
	nabla_b[-1]=delta
	nabla_w[-1]=np.dot(delta,activations[-2].transpose())
	

	for l in range(1,len(params.sizes)-1):
		delta=np.dot(params.weights[-l].transpose(),delta)*(activations[-l-1])*(1-activations[-l-1])
		nabla_b[-l-1]=delta
		nabla_w[-l-1]=np.dot(delta,activations[-l-2].transpose())

	return (nabla_b,nabla_w)

def vectorize(y):
	global params
	a=np.zeros((params.classes,1))
	a[y]=1
	return a 


def train_nn(Xtrain,ytrain,reg,lr,iters,mini_batch_size,layers):
	training_data=zip(Xtrain,ytrain)
	return gradient_descent(training_data,iters,mini_batch_size,lr,reg,layers)

def NN_predict(db,theta,y_val=None):
	X_test=np.array(db)
	X_test=np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)
	activation=X_test
	for w in theta:
		print(activation.shape)
		activation=expit(np.dot(activation,w.transpose()))
		activation=np.concatenate((np.ones((activation.shape[0],1)),activation),axis=1)

	predictions=np.argmax(activation,axis=1)
	if y_val is not None:
		correct=np.sum(predictions==y_val)
		return correct*100/size(y)
	else:
		return predictions.reshape((predictions.shape[0],1))