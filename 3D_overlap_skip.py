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
import random
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
parser.add_argument('--frames', type=int,help='The amount of frames you want to skip when comparing the overlap between frames.')
parser.add_argument('--percent',type=int,help='percentage of gradient points considered')
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

######### HERE WE HAVE THE VALUE THAT DICTATES IF WE USE RANDOM SAMPLING OR NOT
#sampling=True
sampling=False

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
ax=585
ay=585
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
  

def check_around(dpoints,xminus,xplus,yminus,yplus):
    X2=dpoints[i][0]
    Y2=dpoints[i][1]
    for j in range(-xminus,xplus,1):
	for k in range(-yminus,yplus,1):
		if(binmat[X2+j][Y2+k]==1):
			return 1
    return 0


frame_count=args.frames

## make arrays to do skipping of frames
depth_array=np.zeros((frame_count,480,640))
indices_array=np.zeros((frame_count,480*640,2))
poses_array=np.zeros((frame_count,4,4))

#print depth_array[0]
#print indices_array
#print poses_array

IDX=0
match_percentage=0
cm_jet = plt.cm.get_cmap('jet')
for batch_idx, (data, target) in enumerate(loader):
  if batch_idx==0 and sampling==True:
    L=len(loader)
    k=int(L*50/float(100))
#    M=L-frame_count
    sample_list=random.sample(range(frame_count,L+1),k)
    sample_list=np.sort(sample_list)
    L3=len(sample_list)-1
  if CUDA:
    data = data.cuda()
  data_var = Variable(data, requires_grad=True)
  if sampling==True: 
   if batch_idx==sample_list[IDX]:
      ## do the thing
    if IDX==L3:
       # do not increment
       break
    else:
      IDX=IDX+1
   else:
      continue
# gradient and image stuff
  model.zero_grad()
  pose = model(data_var)
  pose.mean().backward()

  act = data_var.grad.data.cpu().numpy()
  act = act.squeeze().transpose((1, 2, 0))
  #img = data[0].cpu().numpy()
  #img = img.transpose((1, 2, 0))
 

 # get pose and depth, works for 7Scenes chess dataset; probably really slow..
  if args.scene=='chess':
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
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
  elif args.scene=='fire':
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
      else:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
  elif args.scene=='heads':
    if not train:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
    else:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
  elif args.scene=='stairs':
    if train:
        if batch_idx<500:
         poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
         depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
        elif batch_idx<1000 and batch_idx>=500:
         poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-500)))
         depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-500)))
        elif batch_idx<1500 and batch_idx>=1000:
         poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
         depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
        else:
         poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-1500)))
         depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-1500)))
    else:
       if batch_idx<500:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
       else:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-500)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-500)))
  elif args.scene=='office':
   if not train:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
   else:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      elif batch_idx<4000 and batch_idx>=3000:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
      elif batch_idx<5000 and batch_idx>=4000:
        poses=np.loadtxt(osp.join(seq_dir[4],'frame-{:06d}.pose.txt'.format(batch_idx-4000)))
        depth=imageio.imread(osp.join(seq_dir[4],'frame-{:06d}.depth.png'.format(batch_idx-4000)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[5],'frame-{:06d}.pose.txt'.format(batch_idx-5000)))
        depth=imageio.imread(osp.join(seq_dir[5],'frame-{:06d}.depth.png'.format(batch_idx-5000)))
  elif args.scene=='pumpkin':
   if not train:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
   else:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      elif batch_idx<4000 and batch_idx>=3000:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
  elif args.scene=='redkitchen':
   if not train:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      elif batch_idx<4000 and batch_idx>=3000:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[4],'frame-{:06d}.pose.txt'.format(batch_idx-4000)))
        depth=imageio.imread(osp.join(seq_dir[4],'frame-{:06d}.depth.png'.format(batch_idx-4000)))
   else:
      if batch_idx<1000:
        poses=np.loadtxt(osp.join(seq_dir[0],'frame-{:06d}.pose.txt'.format(batch_idx)))
        depth=imageio.imread(osp.join(seq_dir[0],'frame-{:06d}.depth.png'.format(batch_idx)))
      elif batch_idx<2000 and batch_idx>=1000:
        poses=np.loadtxt(osp.join(seq_dir[1],'frame-{:06d}.pose.txt'.format(batch_idx-1000)))
        depth=imageio.imread(osp.join(seq_dir[1],'frame-{:06d}.depth.png'.format(batch_idx-1000)))
      elif batch_idx<3000 and batch_idx>=2000:
        poses=np.loadtxt(osp.join(seq_dir[2],'frame-{:06d}.pose.txt'.format(batch_idx-2000)))
        depth=imageio.imread(osp.join(seq_dir[2],'frame-{:06d}.depth.png'.format(batch_idx-2000)))
      elif batch_idx<4000 and batch_idx>=3000:
        poses=np.loadtxt(osp.join(seq_dir[3],'frame-{:06d}.pose.txt'.format(batch_idx-3000)))
        depth=imageio.imread(osp.join(seq_dir[3],'frame-{:06d}.depth.png'.format(batch_idx-3000)))
      elif batch_idx<5000 and batch_idx>=4000:
        poses=np.loadtxt(osp.join(seq_dir[4],'frame-{:06d}.pose.txt'.format(batch_idx-4000)))
        depth=imageio.imread(osp.join(seq_dir[4],'frame-{:06d}.depth.png'.format(batch_idx-4000)))
      elif batch_idx<6000 and batch_idx>=5000:
        poses=np.loadtxt(osp.join(seq_dir[5],'frame-{:06d}.pose.txt'.format(batch_idx-5000)))
        depth=imageio.imread(osp.join(seq_dir[5],'frame-{:06d}.depth.png'.format(batch_idx-5000)))
      else:
        poses=np.loadtxt(osp.join(seq_dir[6],'frame-{:06d}.pose.txt'.format(batch_idx-6000)))
        depth=imageio.imread(osp.join(seq_dir[6],'frame-{:06d}.depth.png'.format(batch_idx-6000)))
 
   
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

  #img *= stats[1]
  #img += stats[0]
  #img *= 255
  #img = img[:, :, ::-1]
  #img = np.clip(img, 0, 255)
## act is the gradients and img is the image data

 ## We want to take every frame except only the first and compare with the previous.
 ## comparison is done by checking how many high gradient points are present in both pictures 
 ## presence is defined if an important point is within a certain pixel distance
  
  # TODO: maybe make into a commmand line argument 
  percentage = args.percent

  normgrad= np.linalg.norm(act, axis=2)

  ## this is the sorted list of image coordinates w.r.t gradient values from smallest to largest
  indices = np.dstack(np.unravel_index(np.argsort(normgrad.ravel()),(480,640)))
   
   # if we are on the first frame we just save the gradient points and pose and continue
  if batch_idx<frame_count:
      poses_array[batch_idx]=poses
      indices_array[batch_idx]=indices
      depth_array[batch_idx]=depth
      continue
  matches=0
  
  #for i in range(0,frame_count):
  # print indices_array[0][i][0]


  L=act.shape[0]*act.shape[1]
  L2=int((L*percentage)/100.)
  #print 'L2'
  #print L2
  

  # here we want to extract the 3D points from the top X% of pixels using the given depth map;
  # to get the 3D point we use the intrinsic parameters to get the 3D point in camera coords,
  # then we use camera to world extrinsic matrix to get to the global coords

  # The same is then done to project the point down into the second frame
 
  dpoints= []
  poseinv=np.linalg.inv(poses) # invert pose matrix of frame 2
  # this is to get from world coords to camera coordinates of frame 2 
  XMAX=act.shape[0]
  YMAX=act.shape[1]
  binmat=np.zeros((XMAX,YMAX))
  for j in range(L-1,L-1-L2,-1):
	binmat[indices[0,j,0]][indices[0,j,1]]=1
  # calculate 3D points from frame 1 then project into frame 2
  for i in range(L-1,L-1-L2,-1):


    Z= float(depth_array[0][int(indices_array[0][i][0])][int(indices_array[0][i][1])])
    if Z<=0 or Z==65535:
     # we dont consider points with depth smaller than or equal to zero 
     # and 65535 are invalid depths according to documentation
     continue

 # calculate 3D point in camera 1 coords
    X=(indices_array[0][i][1]-YMAX/2.-u0)/(ax/Z)
    Y=(indices_array[0][i][0]-XMAX/2.-v0)/(ay/Z)

# we get the 3D world coordinates by multiplying the 3D-point with the pose
    #print np.matmul(poses_prev,[X, Y, Z, 1])

# we now multiply with the inverse of the second pose to get the point in camera 2 coords
    D=np.matmul(poseinv,np.matmul(poses_array[0],[X, Y, Z, 1]))
    X2=int(XMAX/2.+v0+(D[1])*(ay/D[2]))
    Y2=int(YMAX/2.+u0+(D[0])*(ax/D[2]))
    if X2>act.shape[0] or Y2>act.shape[1] or X2<0 or Y2<0:
	continue
# save X,Y in a list
    dpoints.append([X2, Y2] ) ## '(Y,X)' technically since it is how images are indexed
    #print dpoints[0][0]
    #print dpoints[0][1]
    #print dpoints
    #print dpoints
    #print indices[0][i]
    #print indices_prev[0][i]
  
# when all the projected points are known we just compare them to the top X% of gradient points in frame 2 and save the overlap
  pad=3
  xminus=pad
  xplus=pad
  yminus=pad
  yplus=pad
  for i in range(len(dpoints)):
    X2=dpoints[i][0]
    Y2=dpoints[i][1]
    if (X2-xminus)<0:
	xminus=pad+pad-X2
    if X2+xplus>=XMAX:
	xplus=XMAX-X2
    if (Y2-yminus)<0:
	yminus=pad+pad-Y2
    if Y2+yplus>=YMAX:
	yplus=YMAX-Y2
    M=check_around(dpoints,xminus,xplus,yminus,yplus)
    matches=matches+M
    #for j in range(-xminus,xplus,1):
#	for k in range(-yminus,yplus,1):
#		if(binmat[X2+j][Y2+k]==1):
#			matches=matches+1
#			break
  #   X1=indices[0,j,0] 
  #   Y1=indices[0,j,1]
     #if ((X1-X2)**2+(Y1-Y2)**2)<=25:
            #print 'found'
            #print X2
            #print indices2[0,j]
            #print '---'
            #print Y1
            #print Y2
            #print indices[0,i]
      #      matches=matches+1
      #      break          



#  threshold=0 # distance from projected point which the top X% point may lie while counting as corresponding

  # for x in projected_pixels check how large overlap between that and the indices list
  match_percentage=match_percentage+matches/float(L2)

#  for i in range(0,frame_count):
#    print indices_array[i]
#  print '---------------'
 
### change saved indices
  for i in range(0,frame_count-1):
    indices_array[i]=indices_array[i+1]
    poses_array[i]=poses_array[i+1]
    depth_array[i]=depth_array[i+1]
  indices_array[frame_count-1]=indices
  poses_array[frame_count-1]=poses
  depth_array[frame_count-1]=depth

#  for i in range(0,frame_count):
 #   print indices_array[i]

  #sys.exit(-1)



  #print match_percentage
  if batch_idx % 200 == 0:
    print '{:d} / {:d}'.format(batch_idx, len(loader))
#batch_idx=1
result_filename=args.scene+'_{:d}frame.txt'.format(args.frames)#'output.txt'
f=open(result_filename,'a')
if sampling==True:
 print match_percentage/float(L3)
 f.write('{:d}%:'.format(args.percent)+str(match_percentage/float(L3))+'\n')
else:
 print match_percentage/float(batch_idx)
 f.write('{:d}%:'.format(args.percent)+str(match_percentage/float(batch_idx))+'\n')
f.close()
sys.exit(-1)
