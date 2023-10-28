from flask import Flask, request , render_template
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
#from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import *
from torch.utils.data import Dataset, DataLoader

#import matplotlib.pyplot as plt




app = Flask(__name__) 
#print ("hello")


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = models.resnet18(pretrained=True)
# if device == 'cuda':
#     net = net.cuda() 
# net


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

use_cuda = False
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 5)
#net.fc = net.fc.cuda() if use_cuda else net.fc



checkpoint=torch.load('best_checkpoint_resnet18.model')
classes = ['bargraph', 'flowchart', 'linegraph', 'piechart', 'scatterplot']
model=net
model.load_state_dict(checkpoint)
model.eval()


#Transforms
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])


#model = model.to(torch.device('cuda'))


def prediction(img_path,transformer):
    
    image=Image.open(img_path)
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    # if torch.cuda.is_available():
    #     image_tensor.cuda()
        
    #input1=Variable(image_tensor.cuda())
    input1=Variable(image_tensor)
    
    
    output=model(input1)
    
    index=output.data.cpu().numpy().argmax()
    
    pred=classes[index]
    
    return pred
    


@app.route('/')

def welcome():
    return render_template('welcome.html')


@app.route('/predict',methods = ['POST'] )
def predict_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    img = request.files['image']
    final_answer = prediction(img,transformer)
    return final_answer
    



if __name__ == '__main__':
    app.run(debug=True, port = 5000)
