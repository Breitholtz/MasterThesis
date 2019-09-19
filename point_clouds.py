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
import imageio
import skimage.transform as trans
from mpl_toolkits.mplot3d import Axes3D
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
  #transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=stats[1])])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# dataset
train = not args.val
if train:
  print 'TRAIN data'
else:
  print 'VAL data'
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
  transform=data_transform, target_transform=target_transform, seed=seed)
if args.dataset == '7Scenes':
  data_set = SevenScenes(**kwargs)
elif args.dataset == 'RobotCar':
  data_set = RobotCar(**kwargs)
else:
  raise NotImplementedError

# intrinsic camera parameters (for 7Scenes dataset) correct??
u0=0#320
v0=0#240
ax=585.0
ay=585.0
# scale principal point and focal length since we will resized the images; originally they are 640x480 (WxH)
#u0=u0*(341/640.)
#v0=v0*(256/480.)
#ax=ax*(341/640.)
#ay=ay*(256/480.)


# note that we assume no skew
#K = [[ax, 0, u0, 0],[0, ay, v0, 0],[0, 0, 1, 0],[0,0,0,1]]
#print 'intrinsic camera matrix'
#print K
#print 'inverse'
#Kinv =  np.linalg.inv(K)
#print Kinv

# loader (batch_size MUST be 1)
loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=5,
  pin_memory=True)

#
img, _ = data_set[0]

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
  print 'CUDA used'
  torch.cuda.manual_seed(seed)
  model.cuda()

## decide which sequences to use
if train:
    split_file= osp.join(osp.join(osp.expanduser(data_dir),args.scene),'TrainSplit.txt')
else:
    split_file= osp.join(osp.join(osp.expanduser(data_dir),args.scene),'TestSplit.txt')
with open(split_file,'r') as f:
    seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
    
scene_dir=osp.join(data_dir,args.scene)
# directories of the sequences to be used
seq_dir=[osp.join(scene_dir,'seq-{:02d}'.format(i)) for i in seqs]
#print seq_dir
  
match_percentage=0
cm_jet = plt.cm.get_cmap('jet')
for batch_idx, (data, target) in enumerate(loader):
  if CUDA:
    data = data.cuda()
  data_var = Variable(data, requires_grad=True)


# gradient and image stuff
  model.zero_grad()
  pose = model(data_var)
  pose.mean().backward()

  act = data_var.grad.data.cpu().numpy()
  act = act.squeeze().transpose((1, 2, 0))
  img = data[0].cpu().numpy()
  img = img.transpose((1, 2, 0))
 

 # get pose and depth, works for 7Scenes dataset; probably really slow..
  if not train:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
  else:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<=2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<=3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
      
  # resize the depth to be the same as the RGB image
 # depth=trans.resize(depth,(256,341),anti_aliasing=True,mode='reflect')
 # plt.imshow(depth)
 # plt.show()
 # saliency map = data*gradient, comment next line if you want
  # saliency map = gradient
  # act *= img
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
## act is the gradients and img is the image data

 ## We want to take every frame except only the first and compare with the previous.
 ## comparison is done by checking how many high gradient points are present in both pictures 
 ## presence is defined if an important point is within a certain pixel distance
  
  # TODO: maybe make into a commmand line argument 
  percentage = 30

  normgrad= np.linalg.norm(act, axis=2)

  ## this is the sorted list of image coordinates w.r.t gradient values from smallest to largest
  indices = np.dstack(np.unravel_index(np.argsort(normgrad.ravel()),(480,640)))
   
   # if we are on the first frame we just save the gradient points and pose and continue
  if batch_idx==0:
      depth_prev=depth
      poses_prev=poses
      indices_prev=indices
      img_prev=img
      continue
  matches=0
  if batch_idx==1:
      continue
  
 
  L=act.shape[0]*act.shape[1]
  L2=int(np.floor((L*percentage)/100.))
  #print 'L2'
  #print L2


  # here we want to extract the 3D points from the top X% of pixels using the given depth map;
  # to get the 3D point we use the intrinsic parameters to get the 3D point in camera coords,
  # then we use camera to world extrinsic matrix to get to the global coords

  # The same is then done to project the point down into the second frame
 
  points= []
  poseinv=np.linalg.inv(poses) # invert pose matrix of frame 2
  # this is to get from world coords to camera coordinates of frame 2 

  # open file for point cloud
  #f=open('data-{:06d}.ply'.format(batch_idx),'wb')
  # write header
 # f.writelines(['ply \n','format ascii 1.0\n','element vertex {}\n'.format(L2-1),'property float x\n','property float y\n','property float z \n','property uchar red\n','property uchar green\n','property uchar blue\n','end_header\n'])
 
  fig=plt.figure()
  axes=Axes3D(fig)
  
  # calculate 3D points from frame 1 then project into frame 2
  for i in range(L-1,L-1-L2-1,-1):

    colour=img_prev[indices_prev[0][i][0]][indices_prev[0][i][1]]
    Z= float(depth_prev[indices_prev[0][i][0]][indices_prev[0][i][1]])
    if Z<=0:
     # we dont consider points with depth smaller than or equal to zero
     continue


 # calculate 3D point in camera 1 coords
    X=(indices_prev[0][i][1]-img.shape[1]/2.-u0)*Z/ax
    Y=(indices_prev[0][i][0]-img.shape[0]/2.-v0)*Z/ay
    #print poses_prev
    #print [X, Y, Z, 1]
    D=np.matmul(poses_prev,[X, Y, Z, 1])
    D=D/1000.
    axes.scatter(D[0],D[1],D[2])
    #print D
    #f.write('{} {} {} {} {} {}\n'.format(D[0],D[1],D[2],int(colour[0]),int(colour[1]),int(colour[2])))
    #f.write('{} {} {}\n'.format(D[0],D[1],D[2]))
    #points.append([D[0], D[1], D[2]])
    #print 'saved'
    # pymesh.save_mesh("frame-{:06d}.ply".format(batch_idx),points)
  plt.show()
  # here we want to save the 3D points and then put them in PLI format to get a point cloud for each frame
  # we might also want to get the colour of the deprojected pixel to colour the points in the cloud
  #f.close()
  print 'saved mesh'
  sys.exit(-1)
  depth_prev=depth
  img_prev=img
  poses_prev=poses
  indices_prev=indices
sys.exit(-1)
