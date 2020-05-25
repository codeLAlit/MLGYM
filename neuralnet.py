from torch import nn


# THE DIMENSIONALTIY OF THE FIRST LAYER 'S INPUT WILL BE DEPENDANT ON THE INPUT DIMENSIONALITY (input_d) 
# THE DIMENSIONALTIY OF THE LAST LAYER 'S OUTPUT WILL BE DEPENDANT ON THE OUTPUT DIMENSIONALITY (output_d)

class NeuralNet(nn.Module):
	def __init__(self,layer_list,input_d,output_d):
        super().__init__()

        self.final_layer_list=[input_d]
        self.ReLU=nn.ReLU()
        self.layers1=[]
        for q in layer_list :
        	self.final_layer_list.append(q)
        self.final_layer_list.append(output_d)
        for i in range(1,len(final_layer_list)):
        	self.layers1.append(nn.Linear(final_layer_list[i-1], final_layer_list[i]))


       
    def forward(self, x):
        for layer1 in self.layers1:
        	x=layer1(x)
        	x=self.ReLU(x) 

        return x	


def update(x,y,lr):
    opt=optim.Adam(model.parameters(),lr,weight_decay=1e-5)
    y_hat=model(x)
    loss=loss_func(y_hat,y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()
    
