import cv2, math
import numpy as np
from os import listdir, walk
from os.path import isfile, join
from matplotlib import pyplot as plt
import sys
from  PIL  import Image
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim


import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def rotateImage(x1, y1, x2, y2, image):
    angle = math.atan((y2-y1)/(x2-x1))*180/math.pi
    center = (int((x1+x2)/2), int((y1+y2)/2))
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(src = image, M=rotate_matrix, dsize=(image.shape[0], image.shape[1]))
    return rotated_image

def detectFaceEyes(img):
    
    eye1 = None
    eye2 = None
    face = None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    eyect = 0
    facect = 0
    scale_factor = 1.05
    min_neighbors = 4
    while (eyect != 2 or facect != 1) and min_neighbors < 50:
        eye1 = None
        eye2 = None
        face = None

        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        eyect = 0
        facect = len(faces)

        for (x, y, w, h) in faces:
            face = (x,y,w,h)

            roi_gray = gray[y:y+h+1, x:x+w+1]
            roi_color = img[y:y+h+1, x:x+w+1]
            eyes = eye_cascade.detectMultiScale(roi_gray, scale_factor, min_neighbors)
            eyect += len(eyes)
            for(x2,y2,w2,h2) in eyes:
                if eye1 is None:
                    eye1 = (x2,y2,w2,h2)
                else:
                    eye2 = (x2,y2,w2,h2)

                centerx = int(x2+w2/2)
                centery = int(y2+h2/2)


        min_neighbors += 1
    if min_neighbors == 50:
        return ((-1, -1, -1, -1), (-1, -1, -1, -1), (-1, -1, -1, -1))
    return (face, eye1, eye2)
    
def checkValidAndRotate(img):
    tup = detectFaceEyes(img)
    face = tup[0]
    eye1 = tup[1]
    eye2 = tup[2]
    
    if face[0] == -1:
        return False, face, eye1, eye2, None
    return True, face, eye1, eye2, rotateImage(face[0] + eye1[0] + int(eye1[2]/2), face[1] + eye1[1] + int(eye1[3]/2), face[0] + eye2[0] + int(eye2[2]/2), face[1] + eye2[1] + int(eye2[3]/2), img) 
  
def addBorder(img):
    blank_canvas = np.zeros((3*img.shape[0], 3*img.shape[1], 3), dtype=np.uint8)
    blank_canvas.fill(255)
    blank_canvas[img.shape[0]:2*img.shape[0], img.shape[1]:2*img.shape[1]] = img

    return blank_canvas

   
'''
def removeBG(img):
    mask = createMask()
    cv2.imshow("mask", mask)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):
            img[i][j][0] = min(img[i][j][0] + 255 - mask[i][j][0], 255)
            img[i][j][1] = min(img[i][j][1] + 255 - mask[i][j][1], 255)
            img[i][j][2] = min(img[i][j][2] + 255 - mask[i][j][2], 255)
    return img
'''

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def getMask(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)
    return pb_np
    '''
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')
    '''
def createMask():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data')
    prediction_dir = os.path.join(os.getcwd(), 'results/')
    print(prediction_dir)
    model_dir = os.path.join(os.getcwd(), model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
       print("...load U2NEP---4.7 MB")
       net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        #if not os.path.exists(prediction_dir):
        #    os.makedirs(prediction_dir, exist_ok=True)
        del d1,d2,d3,d4,d5,d6,d7
        
        mask = getMask(img_name_list[i_test],pred,prediction_dir)
        img = cv2.imread(img_name_list[0])
        #cv2.imshow("mask", mask)
        #cv2.imshow("img", img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        for i in range (mask.shape[0]):
            for j in range (mask.shape[1]):
                if(img[i][j][0] > mask[i][j][0] or img[i][j][1] > mask[i][j][1] or img[i][j][2] > mask[i][j][2]):
                    img[i][j][0] = 255
                    img[i][j][1] = 255
                    img[i][j][2] = 255
                else:
                    img[i][j][0] = min(img[i][j][0] + 255 - mask[i][j][0], 255)
                    img[i][j][1] = min(img[i][j][1] + 255 - mask[i][j][1], 255)
                    img[i][j][2] = min(img[i][j][2] + 255 - mask[i][j][2], 255)
        return img
        
        


valid, face, eye1, eye2, img = checkValidAndRotate(addBorder(createMask()))
cropped_img = img[int(face[1]- 0.4 * (face[3])):int(face[1] + 1.6 * face[3]), int(face[0]+face[2]/2 - face[3]):int(face[0]+face[2]/2 + face[3])]
resized_img = cv2.resize(cropped_img, (600,600))
cv2.imshow("finalimg", resized_img)
cv2.waitKey()
cv2.destroyAllWindows()


