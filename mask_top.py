"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 ## Modified by Adam Breitholtz 
import set_paths
from models.posenet import PoseNet
from dataset_loaders.seven_scenes import SevenScenes
#from dataset_loaders.robotcar import RobotCar
from common.train import load_state_dict
import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
import configparser
import torch.cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, models
import matplotlib.pyplot as plt


# config
parser = argparse.ArgumentParser(description='Activation visualization script')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--config_file', type=str,
                    help='configuration file used for training')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--val', action='store_true', help='Use val split')
parser.add_argument('--output_dir', type=str, required=True,
  help='Output directory for video')
parser.add_argument('--percent', type=int, required=True, help='How much is not blocked out of the gradient pixels in percent.')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')

# model
feature_extractor = models.resnet34(pretrained=False)
model = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
model.eval()

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
  loc_func = lambda storage, loc: storage
  checkpoint = torch.load(weights_filename, map_location=loc_func)
  load_state_dict(model, checkpoint['model_state_dict'])
  print 'Loaded weights from {:s}'.format(weights_filename)
else:
  print 'Could not load weights from {:s}'.format(weights_filename)
  sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_file = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
# transformer
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=stats[1])])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# dataset
train = not args.val
if train:
  print 'Visualizing TRAIN data'
else:
  print 'Visualizing VAL data'
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
  transform=data_transform, target_transform=target_transform, seed=seed)
if args.dataset == '7Scenes':
  data_set = SevenScenes(**kwargs)
elif args.dataset == 'RobotCar':
  data_set = RobotCar(**kwargs)
else:
  raise NotImplementedError

# loader (batch_size MUST be 1)
loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=5,
  pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
  torch.cuda.manual_seed(seed)
  model.cuda()


print 'Calculating mean colour of dataset'
S=[0,0,0]
for idx, (data, target) in enumerate(loader):

  # calculate mean R,G and B values of the dataset

  # transforming of data
  im=data[0].cpu().numpy()
  im=  im.transpose((1,2,0))
  im *= stats[1]
  im += stats[0]
  im *= 255
  im = im[:, :, ::-1]

  S= S + np.mean(np.mean(im,axis=0),axis=0)
  
print S/idx
MEAN=np.around(S/idx)
## TODO: this RGB mean part takes 24s currently which frankly is disgustingly slow. Make it faster!

## get the sequences that are used

if train:
    split_file=osp.join(osp.join(osp.expanduser(data_dir),args.scene),'TrainSplit.txt')
else:
    split_file=osp.join(osp.join(osp.expanduser(data_dir),args.scene),'TestSplit.txt')
with open(split_file,'r') as f:
    seqs =[int(l.split('sequence')[-1])for l in f if not l.startswith('#')]
scene_dir=osp.join(args.output_dir,args.scene)
seq_dir=[osp.join(scene_dir,'seq-{:02d}'.format(i)) for i in seqs]
# inference
cm_jet = plt.cm.get_cmap('jet')
for batch_idx, (data, target) in enumerate(loader):
  if CUDA:
    data = data.cuda()
  data_var = Variable(data, requires_grad=True)
  

  model.zero_grad()
  pose = model(data_var)
  pose.mean().backward()

  act = data_var.grad.data.cpu().numpy()
  act = act.squeeze().transpose((1, 2, 0))
  img = data[0].cpu().numpy()
  img = img.transpose((1, 2, 0))

  # saliency map = data*gradient, comment next line if you want
  # saliency map = gradient
  #act *= img
  act = np.amax(np.abs(act), axis=2)
  act -= act.min()
  act /= act.max()
  act = cm_jet(act)[:, :, :3]
  act *= 255

  img *= stats[1]
  img += stats[0]
  img *= 255
  img = img[:, :, ::-1]
  img = np.clip(img, 0, 255)


  # img = 0.5*img + 0.5*act
  # img = np.clip(img, 0, 255)

## act is the gradients and img is the image data
 





 ### here we want to find the N most salient pixels and then fill the pixel 
 ### with the mean colour of the dataset and save or use the new image somewhere
 
  # percentage we want to 'grey out'
  # TODO: maybe make into a commmand line argument 
  percentage = args.percent

  # we sort the image values and find the value of the smallest included element
  # then we search the gradient image and grey the pixels in the corresponding image 
  # rewrite this trash pls 
  normgrad= np.linalg.norm(act, axis=2)
  # sort gradient values
  sort = np.sort(normgrad,axis=None)
  ## this indices is the sorted list of image coordinates w.r.t gradient values from smallest to largest
  indices = np.dstack(np.unravel_index(np.argsort(normgrad.ravel()),(256,341)))
 # print normgrad
 # print indices.shape
 # print normgrad[indices[0,1,0],indices[0,1,1]]
 # print normgrad[indices[0,10000,0],indices[0,10000,1]]




  # find the lowest value which is included in the percentage
  L=len(sort)
  L2=np.floor((L*percentage)/100)
   # search through gradient image and grey out pixels in real image if the gradient value is above the threshold
  newimg=img
  # plot the old image
  # plt.imshow(img.astype(int))
  # plt.show()
  count=0

  ### here we want to find which sequence of which dataset and modify the filename so we put the images directly in the correct directory after manipulation
  if args.scene=='chess':
      if not train:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        else:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
      else:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<2000 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<3000 and batch_idx>=2000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-2000))
        elif batch_idx<4000 and batch_idx>=3000:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-3000))
  elif args.scene=='fire':
      if not train:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        else:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
      else:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        else:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
  elif args.scene=='office':
      if not train:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<2000 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<3000 and batch_idx>=2000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-2000))
        elif batch_idx<4000 and batch_idx>=3000:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-3000))
      
      else:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<2000 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<3000 and batch_idx>=2000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-2000))
        elif batch_idx<4000 and batch_idx>=3000:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-3000))
        elif batch_idx<5000 and batch_idx>=4000:
            out_filename = osp.join(seq_dir[4], 'frame-{:06d}.color.png'. format(batch_idx-4000))
        elif batch_idx<6000 and batch_idx>=5000:
            out_filename = osp.join(seq_dir[5], 'frame-{:06d}.color.png'. format(batch_idx-5000))
  elif args.scene=='redkitchen':
      if not train:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<2000 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<3000 and batch_idx>=2000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-2000))
        elif batch_idx<4000 and batch_idx>=3000:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-3000))
        elif batch_idx<5000 and batch_idx>=4000:
            out_filename = osp.join(seq_dir[4], 'frame-{:06d}.color.png'. format(batch_idx-4000))
      else:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<2000 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<3000 and batch_idx>=2000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-2000))
        elif batch_idx<4000 and batch_idx>=3000:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-3000))
        elif batch_idx<5000 and batch_idx>=4000:
            out_filename = osp.join(seq_dir[4], 'frame-{:06d}.color.png'. format(batch_idx-4000))
        elif batch_idx<6000 and batch_idx>=5000:
            out_filename = osp.join(seq_dir[5], 'frame-{:06d}.color.png'. format(batch_idx-5000))
        elif batch_idx<7000 and batch_idx>=6000:
            out_filename = osp.join(seq_dir[6], 'frame-{:06d}.color.png'. format(batch_idx-6000))
  elif args.scene=='heads':
      if not train:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
      else:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
  elif args.scene=='pumpkin':
      if not train:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        else:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
      else:
        if batch_idx<1000:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<2000 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<3000 and batch_idx>=2000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-2000))
        elif batch_idx<4000 and batch_idx>=3000:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-3000))
  elif args.scene=='stairs':
      if not train:
        if batch_idx<500:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        else:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-500))
      else:
        if batch_idx<500:
            out_filename = osp.join(seq_dir[0], 'frame-{:06d}.color.png'. format(batch_idx))
        elif batch_idx<1000 and batch_idx>=500:
            out_filename = osp.join(seq_dir[1], 'frame-{:06d}.color.png'. format(batch_idx-500))
        elif batch_idx<1500 and batch_idx>=1000:
            out_filename = osp.join(seq_dir[2], 'frame-{:06d}.color.png'. format(batch_idx-1000))
        elif batch_idx<2000 and batch_idx>=1500:
            out_filename = osp.join(seq_dir[3], 'frame-{:06d}.color.png'. format(batch_idx-1500))
    
 # MEAN= np.mean(np.mean(img,axis=0),axis=0)
  

  #out_filename = osp.join(args.output_dir, 'frame-{:06d}.color.png'. format(batch_idx))

# make the non important pixels be greyed out
  for i in range(L-1,int(L-L2-1),-1):
      newimg[indices[0,i,0],indices[0,i,1]]=MEAN
        
  cv2.imwrite(out_filename,newimg)
 # print '-------IMAGE DATA---------'
 # print img[10,10] 
 # print '--------NORM---------'
 # print np.linalg.norm(img[10,10])
 # print '--------END---------'
  if batch_idx % 200 == 0:
    print '{:d} / {:d}'.format(batch_idx, len(loader))

sys.exit(-1)
  
#vwrite.release()
#print '{:s} written'.format(out_filename)
