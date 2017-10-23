'''He Wei Oct23 2017'''

import pickle
import argparse

import os
from social_utils import SocialDataLoader

import numpy as np

import cv2
from PIL import Image, ImageDraw

green = (0,255,0)
red = (0,0,255)


parser = argparse.ArgumentParser()
# Observed length of the trajectory parameter
parser.add_argument('--obs_length', type=int, default=6,
                    help='Observed length of the trajectory')
# Predicted length of the trajectory parameter
parser.add_argument('--pred_length', type=int, default=6,
                    help='Predicted length of the trajectory')
# Test dataset
parser.add_argument('--visual_dataset', type=int, default=1,
                    help='Dataset to be tested on')

# Model to be loaded
parser.add_argument('--epoch', type=int, default=39,
                    help='Epoch of model to be loaded')

# Parse the parameters
sample_args = parser.parse_args()

'''KITTI Training Setting'''

#save_directory = '/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingKITTI-13NTestonKITTI-17/save'
save_directory = '/home/hesl/PycharmProjects/social-lstm-tf-HW/save'

#save_directory ='/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingETH1TestETH0/save/'

with open(os.path.join(save_directory, 'social_config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

#f = open('/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingKITTI-13NTestonKITTI-17/save/social_results.pkl', 'rb')
#f = open('/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingETH1TestETH0/save/social_results.pkl', 'rb')
f = open('/home/hesl/PycharmProjects/social-lstm-tf-HW/save/social_results.pkl', 'rb')
results = pickle.load(f)

dataset = [sample_args.visual_dataset]
data_loader = SocialDataLoader(1, sample_args.pred_length + sample_args.obs_length, saved_args.maxNumPeds, dataset, True, infer=True)

eth_H=np.loadtxt('/home/hesl/PycharmProjects/social-lstm-tf-HW/data/eth/univ/H.txt')
hotel_H=np.loadtxt('/home/hesl/PycharmProjects/social-lstm-tf-HW/data/eth/hotel/H.txt')

#[7,16]
#print(data_loader.data[0][0].shape)

# '''Visualize Ground Truth'''
# for j in range(len(data_loader.frameList[0])):
#
#     # sourceFileName = "/home/hesl/PycharmProjects/social-lstm-tf-HW/data/KITTI-17/img1/" + str(j + 1).zfill(6) + ".jpg"
#
#     sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/hotel/frame-" + str(int(data_loader.frameList[0][j])).zfill(3) + ".jpg"
#     print(sourceFileName)
#
#     avatar= cv2.imread(sourceFileName)
#     #drawAvatar= ImageDraw.Draw(avatar)
#     #print(avatar.shape)
#     xSize  = avatar.shape[1]
#     ySize = avatar.shape[0]
#     #print(data_loader.data[0][0][0])
#     for i in range(data_loader.maxNumPeds):
#          #print(i)
#
#          y=int(data_loader.data[0][j][i][2])
#          x=int(data_loader.data[0][j][i][1])
#          if x!=0 and y!=0:
#              print(x, y)
#
#          cv2.rectangle(avatar, (x  - 2, y  - 2), (x  + 2, y + 2), green,thickness=-1)
#          #drawAvatar.rectangle([(x  - 2, y  - 2), (x  + 2, y + 2)], fill=(255, 100, 0))
#
#     #drawAvatar.rectangle([(466, 139), (91 + 466, 139 + 193.68)])
#     #avatar.show()
#     cv2.imshow("avatar", avatar)
#     cv2.waitKey(0)


print(results[0][1][0][2])



#Visualize result of image coordinate data
#Each Frame
# for k in range(int(len(data_loader.frameList[0])/(sample_args.obs_length+sample_args.pred_length))):
#     #Each
#     for j in range(sample_args.obs_length+sample_args.pred_length):
#
#         sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/eth/frame-" + str(int(data_loader.frameList[0][j+k*(sample_args.obs_length+sample_args.pred_length)])).zfill(3) + ".jpeg"
#         print(sourceFileName)
#
#         avatar= cv2.imread(sourceFileName)
#
#
#         #width
#         xSize  = avatar.shape[1]
#         #height
#         ySize = avatar.shape[0]
#
#
#         for i in range(data_loader.maxNumPeds):
#             if results[k][1][j][i][0] != 0:
#                 # Predicted
#                 yp = int(np.round(results[k][1][j][i][2] * ySize))
#                 xp = int(np.round(results[k][1][j][i][1] * xSize))
#                 cv2.rectangle(avatar, (xp - 2, yp - 2), (xp + 2, yp + 2), red, thickness=-1)
#
#             if results[k][0][j][i][0]!=0:
#                 # GT
#                 y = int(np.round(results[k][0][j][i][2] * ySize))
#                 x = int(np.round(results[k][0][j][i][1] * xSize))
#                 cv2.rectangle(avatar, (x - 2, y - 2), (x + 2, y + 2), green, thickness=-1)
#
#                 print('x,y')
#                 print(x,y)
#             if results[k][1][j][i][0] != 0 and results[k][0][j][i][0]!=0 and results[k][0][j][i][0]==results[k][1][j][i][0]:
#                 cv2.line(avatar, (x, y), (xp, yp), (255,0,0),1)
#
#         cv2.imshow("avatar", avatar)
#         #imagename='/home/hesl/PycharmProjects/social-lstm-tf-HW/plot/visualize-'+str(int(data_loader.frameList[0][j+k*(sample_args.obs_length+sample_args.pred_length)])).zfill(3)+'.png'
#         #cv2.imwrite(imagename, avatar)
#         cv2.waitKey(0)

#Visualize result of world coordinate data
#Each Frame
for k in range(int(len(data_loader.frameList[0])/(sample_args.obs_length+sample_args.pred_length))):
    #Each
    for j in range(sample_args.obs_length+sample_args.pred_length):

        sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/hotel/frame-" + str(int(data_loader.frameList[0][j+k*(sample_args.obs_length+sample_args.pred_length)])).zfill(3) + ".jpg"

        avatar= cv2.imread(sourceFileName)

        xSize  = avatar.shape[1]
        ySize = avatar.shape[0]
        print(sourceFileName)

        for i in range(data_loader.maxNumPeds):


            if results[k][1][j][i][0] != 0:
                pos =np.ones(3)

                # Predicted
                xp = results[k][1][j][i][1]
                yp = results[k][1][j][i][2]

                #pos:[x,y,1]
                pos[0] = xp
                pos[1] = yp

                # pos = np.dot(posp,np.linalg.inv(eth_H.transpose()))
                pos = np.dot(pos, np.linalg.inv(hotel_H.transpose()))

                vp=int(np.around(pos[0]/pos[2]))
                up=int(np.around(pos[1]/pos[2]))

                cv2.rectangle(avatar, (up - 2, vp - 2), (up + 2, vp + 2), red, thickness=-1)


            if results[k][0][j][i][0]!=0:
                # GT
                posp = np.ones(3)

                x= results[k][0][j][i][1]
                y = results[k][0][j][i][2]



                posp[0]=x
                posp[1]=y

                #posp = np.dot(posp,np.linalg.inv(eth_H.transpose()))
                posp = np.dot(posp, np.linalg.inv(hotel_H.transpose()))

                v = int(np.around(posp[0] / posp[2]))
                u = int(np.around(posp[1] / posp[2]))

                print(posp[0] / posp[2],posp[1] / posp[2])

                print('uv:')
                print(u, v)

                cv2.rectangle(avatar, (u - 2, v - 2), (u + 2, v + 2), green, thickness=-1)

            if results[k][1][j][i][0] != 0 and results[k][0][j][i][0]!=0 and results[k][0][j][i][0]==results[k][1][j][i][0]:
                cv2.line(avatar, (u, v), (up, vp), (255,0,0),1)

        cv2.imshow("avatar", avatar)
        imagename='/home/hesl/PycharmProjects/social-lstm-tf-HW/plot/visualize-'+str(int(data_loader.frameList[0][j+k*(sample_args.obs_length+sample_args.pred_length)])).zfill(3)+'.png'
        cv2.imwrite(imagename, avatar)
        #cv2.waitKey(0)


print(len(results))