import matplotlib as mpl
mpl.use('Agg')

import os
import pdb
import torch
import random
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

class VAutoencoder(nn.Module):
    def __init__(self,z_dim):
        super(VAutoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
                        
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
                        
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
                        
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),)
                    
        self.fc_latent = nn.Linear(128*8*8,z_dim)
        self.fc_var = nn.Linear(z_dim,z_dim)
        self.fc_mean = nn.Linear(z_dim,z_dim)

        self.fc_in = nn.Linear(z_dim,128*8*8)

        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64,6,2,2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,32,6,2,2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32,16,6,2,2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16,3,6,2,2),
            nn.Sigmoid()
        )

    def reparametarize(self,mu,var):
          
          std = var.mul(0.5).exp_()
         
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)

    def bottleneck(self, h):
        h = self.fc_latent(h)
        mu, logvar = self.fc_mean(h), self.fc_var(h)
        z = self.reparametarize(mu, logvar)
        
        z_ = z[0:1,:]
        for i in range(x.shape[0]-1):	
          z1 = z[i:i+1,:]
          z2 = z[i+1:i+2,:]
          
          for i in range(100):
            z_ = torch.cat((z_,(1-(i+1)*0.01)*z1+(i+1)*0.01*z2))
        return z_, mu, logvar

    def forward(self,x):
        input_shape = x.shape
        x = self.encoder(x)
        
        x = x.view(x.shape[0],-1)
        
        z,mu,logvar = self.bottleneck(x)
        x = self.fc_in(z)
        x = x.view(x.shape[0],128,8,8)
        
        x = self.decoder(x)
        
        return x,z

class Data_loader(Dataset):
    def __init__(self,files,transform):
      self.transform = transform
      self.filenames = files

    def __getitem__(self,index):
      filename = self.filenames[index]
    
      with open(filename,'rb') as f:
        image = Image.open(f).convert('RGB')
      image = self.transform(image)
      return image
  
    def __len__(self):
      return len(self.filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--svd_model', required=False,   help="save inference", default='"saved_model/pretrained.pth"', type=str)
    parser.add_argument('--src_dir', required=True,   help="save inference", default='data/', type=str)
    parser.add_argument('--latent_space', required=False,   help="latent_space", default=1024, type=int)

    args   = parser.parse_args()

    print(args)

    if not os.path.exists(args.src_dir):
        os.makedirs(args.src_dir)
    
    dataset_path = 'interp/'

    classes_names = os.listdir(dataset_path)

    input_transforms = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()])

    files = []
    for i in range(len(classes_names)):
      class_name = classes_names[i]
      path = dataset_path+class_name + "/"
      file_names = os.listdir(path)
      for file_name in file_names:
        file_path = os.path.join(path,file_name)
        files.append(file_path)

    np.random.shuffle(files)


    train_files = files[:int(0.9*len(files))]

    test_files = files[int(0.9*len(files)):]

    train_dataset = Data_loader(train_files,input_transforms)
    test_dataset = Data_loader(test_files,input_transforms)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    model = VAutoencoder(args.latent_space)

    def load_my_state_dict(model, state_dict):
            
            own_state = model.state_dict()
            
            for name, param in state_dict.items():
                if name not in own_state:
                    print("[weight not copied for %s]"%(name)) 
                    continue
                #print(name)
                own_state[name].copy_(param)
            return model
    
    model = load_my_state_dict(model,torch.load("saved_model/pretrained.pth"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for step, (x) in enumerate(train_loader):

      x1 = x.to(device).float()

      output,latent = model(x1)

      image = output.detach().to('cpu')
      comparison = image

      if not os.path.exists(args.src_dir+str(i)+'/'):
        os.makedirs(args.src_dir+str(step)+'/')
      
      for i in range(comparison.shape[0]):
        save_image(comparison[i],args.src_dir+str(step) +'/reconstruction_latent%04d.png'%(i), nrow=11)

        
