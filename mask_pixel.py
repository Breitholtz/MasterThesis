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

# opencv init
 ##fourcc = cv2.VideoWriter_fourcc(*'XVID')
#model_name = 'posenet' if args.weights.find('posenet') >= 0 else 'vidvo'
#out_filename = osp.join(args.output_dir, '{:s}_{:s}_attention_{:s}.avi'.
#                        format(args.dataset, args.scene, model_name))
# get frame size
#img, _ = data_set[0]
#vwrite = cv2.VideoWriter(out_filename, fourcc=fourcc, fps=20.0,
                         #frameSize=(img.size(2), img.size(1)))
#print 'Initialized VideoWriter to {:s} with frames size {:d} x {:d}'.\
 # format(out_filename, img.size(2), img.size(1))
'''
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
'''

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
  act *= img
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
  percentage = 10

  # we sort the image values and find the value of the smallest included element
  # then we search the gradient image and grey the pixels in the corresponding image 
  # rewrite this trash pls 
  normgrad= np.linalg.norm(act, axis=2)
  # sort gradient values
  sort = np.sort(normgrad,axis=None)
  ## this indices is the sorted list of image coordinates w.r.t gradient values from smallest to largest
  indices = np.dstack(np.unravel_index(np.argsort(normgrad.ravel()),(256,341)))
  print normgrad
  print indices.shape
  print normgrad[indices[0,1,0],indices[0,1,1]]
  print normgrad[indices[0,10000,0],indices[0,10000,1]]



  sys.exit(-1)

  # find the lowest value which is included in the percentage
  L=len(sort)
  L2=np.floor((L*percentage)/100)
  M= sort[int(L-L2)]
   # search through gradient image and grey out pixels in real image if the gradient value is above the threshold
  newimg=img
  # plot the old image
  # plt.imshow(img.astype(int))
  # plt.show()
  count=0
  out_filename = osp.join(args.output_dir, '{:s}_{:s}_masked_{:d}.png'. format(args.dataset, args.scene, batch_idx))

  for i in range(1,256):
      for j in range(1,341):
        if normgrad[i,j]>=M:
            # make sure that we dont replace too many pixels if there are several with the same values
            if count<int(L2): 
                newimg[i,j]=MEAN
                count=count+1
# plot the new image
#  plt.imshow(newimg.astype(int))
 # plt.show()
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
