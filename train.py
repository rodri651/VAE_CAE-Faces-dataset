
import matplotlib as mpl
mpl.use('Agg')

import os
import pdb
import PIL
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader, Dataset

class VAutoencoder(nn.Module):
  def __init__(self,z_dim):
      super(VAutoencoder,self).__init__()

      self.encoder = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3,stride=2,padding=1),
          nn.ReLU(True),
          #nn.BatchNorm2d(16),  #64x64
          
          nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
          nn.ReLU(True),
          #nn.BatchNorm2d(32), # 32x32
          
          nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
          nn.ReLU(True),
          #nn.BatchNorm2d(64), # 16x16
          
          nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
          nn.ReLU(True),)
          #nn.BatchNorm2d(128)) # 8x8
      
      self.fc_latent = nn.Linear(128*8*8,z_dim)
      self.fc_var = nn.Linear(z_dim,z_dim)
      self.fc_mean = nn.Linear(z_dim,z_dim)

      self.fc_in = nn.Linear(z_dim,128*8*8)

      
      
      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(128,64,6,2,2),
          nn.ReLU(True),
          #nn.BatchNorm2d(64),
          nn.ConvTranspose2d(64,32,6,2,2),
          nn.ReLU(True),
          #nn.BatchNorm2d(32),
          nn.ConvTranspose2d(32,16,6,2,2),
          nn.ReLU(),
          #nn.BatchNorm2d(16),
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
      return z, mu, logvar

  def forward(self,x):
      input_shape = x.shape
      x = self.encoder(x)
      
      x = x.view(x.shape[0],-1)
      
      x,mu,logvar = self.bottleneck(x)
      x = self.fc_in(x)
      x = x.view(input_shape[0],128,8,8)
      
      x = self.decoder(x)
      
      return x,mu,logvar


class UTZappos_loader(Dataset):
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
  dataset_path = 'faces_dataset/'
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

  train_dataset = UTZappos_loader(train_files,input_transforms)
  test_dataset = UTZappos_loader(test_files,input_transforms)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

  parser = argparse.ArgumentParser()
  parser.add_argument('--src_dir', required=True,   help="trial_des", default='data/', type=str)
  parser.add_argument('--latent_space', required=True,   help="latent_space", default=1024, type=int)
  parser.add_argument('--batch_size', required=True,   help="batch size", default=32, type=int)
  parser.add_argument('--lr', required=True,   help="learning rate", default=0.0001, type=float)


  args   = parser.parse_args()

  print(args)

  if not os.path.exists(args.src_dir):
      os.makedirs(args.src_dir)


  model = VAutoencoder(args.latent_space)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  loss_mse = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  EPOCH = 1000
  lambda1 = lambda epoch: pow((1-((epoch-1)/EPOCH)),0.95)
  scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

  all_bce_losses = []
  all_kld_losses = []




  for epoch in range(EPOCH):
      n_correct=0
      n_wrong =0
      losses = []
      bce_losses = []
      kld_losses = []
      for param_group in optimizer.param_groups:
        print(param_group['lr'])
      scheduler.step(epoch)
      
      for step, (x) in enumerate(train_loader):

        x1 = x.to(device).float()
        optimizer.zero_grad()
        output,mean,logvar = model(x1)
        
        loss_output_mse = loss_mse(output, x1)
        
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())/(mean.shape[0]*mean.shape[1])
        
        
        net_loss = loss_output_mse + KLD
        aa = net_loss.detach().to('cpu').numpy()

        net_loss.backward()
        optimizer.step()
        losses.append(aa)
        kld_losses.append(KLD.item())
        bce_losses.append(loss_output_mse.item())
        
      print(epoch,np.mean(np.array(losses)))
      print(epoch,np.mean(np.array(bce_losses)))
      print(epoch,np.mean(np.array(kld_losses)))

      all_bce_losses.append(np.mean(np.array(bce_losses)))
      all_kld_losses.append(np.mean(np.array(kld_losses)))

      if epoch%20 == 0:
        torch.save(model.state_dict(),args.src_dir+str(epoch)+".pth")
      image = output.detach().to('cpu')

      comparison = torch.cat([image[:8], x1[:8].detach().to('cpu')])
      save_image(comparison,args.src_dir+'reconstruction_' + str(epoch) + '.png', nrow=8)
      
      if epoch%20 == 0:
        plt.figure()
        plt.plot(all_bce_losses)
        plt.plot(all_kld_losses)
        plt.savefig(args.src_dir+str(epoch)+".png")
