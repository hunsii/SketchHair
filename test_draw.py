import sys
sys.path.insert(0, "SketchHairSalon")

import cv2
import numpy as np
import datetime
import time
from tkinter import colorchooser
from tkinter import Tk
import os
import argparse

import torch
from models.Unet_At_Bg import UnetAtGenerator, UnetAtBgGenerator
import torchvision.transforms.functional as tf

class Sk2Matte:
    def __init__(self,load_path="./SketchHairSalon/checkpoints/S2M/200_net_G.pth"):
        self.model = UnetAtGenerator(1,1,8,64,use_dropout=True)
        self.device = torch.device('cuda:0')
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        model_name = '/'.join(load_path.split('/')[2:])
        print("Model Sk2Matte (%s) is loaded."%model_name)
    
    def getResult(self,inputs,img=None):
        inputs_tensor = tf.to_tensor(inputs[:,:,np.newaxis])*2.0-1.0
        inputs_tensor = inputs_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            result_tensor = self.model(inputs_tensor)
            result = ((result_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")[...,0]
            # result = util.tensor2im(result_tensor)
            result[result>250] = 255

        return result

class Sk2Image:
    def __init__(self,load_path="./SketchHairSalon/checkpoints/S2I_unbraid/200_net_G.pth"):
        self.model = UnetAtBgGenerator(3,3,8,64,use_dropout=True)
        self.device = torch.device('cuda:0')
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        model_name = '/'.join(load_path.split('/')[2:])
        print("Model Sk2Image (%s) is loaded."%model_name)
    
    def getResult(self,inputs,img,matte, thred=1):
            
        h,w = img.shape[:2]
        noise = self.generate_noise(w,h)

        N = tf.to_tensor(noise)*2.0-1.0
        N = N.unsqueeze(0).to(self.device)

        img_tensor = tf.to_tensor(img)*2.0-1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        M = tf.to_tensor(matte).unsqueeze(0).to(self.device)

        inputs = tf.to_tensor(inputs)*2.0-1.0
        inputs = inputs.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            result_tensor, hair_matte_tensor = self.model(inputs,img_tensor, M,N, thred=thred)
            result = ((result_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")
            hair_matte = ((hair_matte_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")
        return result, hair_matte
    
    def generate_noise(self, width, height):
        weight = 1.0
        weightSum = 0.0
        noise = np.zeros((height, width, 3)).astype(np.float32)
        while width >= 8 and height >= 8:
            noise += cv2.resize(np.random.normal(loc = 0.5, scale = 0.25, size = (int(height), int(width), 3)), dsize = (noise.shape[0], noise.shape[1])) * weight
            weightSum += weight
            width //= 2
            height //= 2
        return noise / weightSum
    
# make output dirs
os.makedirs(f'output/matte/', exist_ok=True)
os.makedirs(f'output/sketch1/', exist_ok=True)
os.makedirs(f'output/sketch2/', exist_ok=True)
os.makedirs(f'output/sketch/', exist_ok=True)
os.makedirs(f'output/face/', exist_ok=True)
os.makedirs(f'output/hair/', exist_ok=True)

# default parameter for text
WINDOW_NAME = 'painter'
ALPHA = 0.25
THICKNESS = 2
GRAY_COLOR = 0
DELAY = 0.25            # S2I model is too heavy. this is display rate

READY_TO_S2I = False

drawing = False # true if mouse is pressed
checkpoint = False

pt1_x , pt1_y = None , None
r, g, b = None, None, None
before_img, before_sketch = None, None

original_backup_list = []
sketch_backup_list = []

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing
    global r,g,b
    global checkpoint
    global original_backup_list
    global sketch_backup_list
    global before_img
    global before_sketch

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x,pt1_y=x,y
        before_img = img.copy()
        before_sketch = sketch_img.copy()
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(b,g,r),thickness=THICKNESS)
            cv2.line(sketch_img,(pt1_x,pt1_y),(x,y),color=(b,g,r),thickness=THICKNESS)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(b,g,r),thickness=THICKNESS)
        cv2.line(sketch_img,(pt1_x,pt1_y),(x,y),color=(b,g,r),thickness=THICKNESS)
        original_backup_list.append(before_img)
        sketch_backup_list.append(before_sketch)
        drawing = False
        checkpoint = True

def onChange(pos):
    pass


white_img = np.zeros((512,512,3), np.uint8)
white_img[:, :, :] = 255
sketch_img = np.zeros_like(white_img)
sketch_img[:, :, :] = GRAY_COLOR
matte = np.zeros_like(sketch_img)

# img_path = 'input/SUMIN.png'
img_path = 'test_data/mapper_res/03055.png'
basename = os.path.basename(img_path)
# img_name = os.path.basename(img_path).split('.')[0]
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (512, 512))
original_img = img.copy()
# bald_img = cv2.resize(cv2.imread('input/SUMIN_bald.png'), (512,512))
cv2.imshow('original_img', img)
result = img.copy()

img = cv2.addWeighted(img, ALPHA, white_img, 1 - ALPHA, 0)
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께
cv2.putText(img, "q: exit", (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
cv2.putText(img, "r: reset", (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
cv2.putText(img, "s: save", (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
cv2.putText(img, "z: undo", (360, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
cv2.putText(img, "c: color", (360, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
original_backup_list.append(img.copy())
sketch_backup_list.append(sketch_img.copy())

S2M = Sk2Matte()
'''
# unbraid
S2I = Sk2Image("./SketchHairSalon/checkpoints/S2I_unbraid/200_net_G.pth")
# braid
S2I = Sk2Image("./SketchHairSalon/checkpoints/S2I_braid/400_net_G.pth")
'''
S2I = Sk2Image("./SketchHairSalon/checkpoints/S2I_unbraid/200_net_G.pth") # unbraid
S2M.getResult(cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)) # fast load

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,line_drawing)
cv2.createTrackbar("R", WINDOW_NAME, 0, 255, onChange)
cv2.createTrackbar("G", WINDOW_NAME, 0, 255, onChange)
cv2.createTrackbar("B", WINDOW_NAME, 0, 255, onChange)
cv2.setTrackbarPos("R", WINDOW_NAME, 255)
cv2.setTrackbarPos("G", WINDOW_NAME, 255)
cv2.setTrackbarPos("B", WINDOW_NAME, 255)

cv2.namedWindow('S2I')
cv2.createTrackbar("Background", 'S2I', 0, 100, onChange)
cv2.setTrackbarPos("Background", 'S2I', 100)
cv2.createTrackbar("Matte", 'S2I', 0, 100, onChange)
cv2.setTrackbarPos("Matte", 'S2I', 0)

before_thred = 1
before_matte_thred = 0
start_time = time.time()

def cvtSketchForS2M(sketched):
    sketch2 = sketched.copy()
    sketch2 = cv2.cvtColor(sketch2, cv2.COLOR_BGR2GRAY)
    sketch2 = np.where(sketch2 == 0, 127, 255).astype(np.uint8)
    return sketch2

while(True):
    r = cv2.getTrackbarPos("R", WINDOW_NAME)
    g = cv2.getTrackbarPos("G", WINDOW_NAME)
    b = cv2.getTrackbarPos("B", WINDOW_NAME)
    thred = cv2.getTrackbarPos("Background", 'S2I') / 100
    matte_thred = cv2.getTrackbarPos("Matte", 'S2I') / 100
    if before_thred != thred and len(original_backup_list) > 1:
        READY_TO_S2I = True
    elif before_matte_thred != matte_thred and len(original_backup_list) > 1:
        READY_TO_S2I = True
    


    cv2.imshow(WINDOW_NAME, img)
    # cv2.imshow('sketch for S2M', cvtSketchForS2M(sketch_img))
    # cv2.imshow('sketch for S2I', sketch_img)
    # cv2.imshow('S2M(matte)', matte)
    cv2.imshow('S2I', result)
    
    input_key = cv2.waitKey(1)
    if input_key == ord('q'): # quit
        break
    elif input_key == ord('r'): # reset
        img = original_backup_list[0].copy()
        matte = np.zeros_like(sketch_img)
        result = original_img

        sketch_img = sketch_backup_list[0].copy()
        sketch_img[:, :, :] = GRAY_COLOR
        start_time = time.time()
        while len(original_backup_list) > 1:
            original_backup_list.pop()
            sketch_backup_list.pop()        
    elif input_key == ord('s'): # save

        # filename = f'{img_name}_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}.jpg'
        # filename = img_name
        cv2.imwrite(f"output/face/{basename}", original_img)
        cv2.imwrite(f"output/sketch1/{basename}", sketch_img)
        cv2.imwrite(f"output/sketch2/{basename}", cvtSketchForS2M(sketch_img))
        cv2.imwrite(f"output/matte/{basename}", matte)
        cv2.imwrite(f"output/sketch/{basename}", result)
        cv2.imwrite(f"test_data/origin_bald/{basename}", cv2.resize(result, (1024,1024)))
        cv2.imwrite(f"output/hair/{basename}", only_hair)
        # cv2.imwrite(f"{filename}", cv2.resize(result, (1024, 1024)))

    elif input_key == ord('z'): # undo
        if len(original_backup_list) > 1:
            img = original_backup_list.pop()
            sketch_img = sketch_backup_list.pop()
        else:
            img = original_backup_list[0]
            sketch_img = sketch_backup_list[0]
        checkpoint = True
    elif input_key == ord('c'): # color pick from palette
        try:
            color_code = colorchooser.askcolor(title = "Choose color",)
            root = Tk()
            root.destroy()
            # print(color_code) # ((R, G, B), code)
            cv2.setTrackbarPos("R", WINDOW_NAME, color_code[0][0])
            cv2.setTrackbarPos("G", WINDOW_NAME, color_code[0][1])
            cv2.setTrackbarPos("B", WINDOW_NAME, color_code[0][2])
        except:
            pass
    elif input_key == ord('i'): # color pick from palette
        pass
        '''
        sk_matte = np.array(cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR))
        sk_rgb = cv2.cvtColor(sketch_img,cv2.COLOR_BGR2RGB)
        sk_gray = cv2.cvtColor(sketch_img,cv2.COLOR_BGR2GRAY)
        sk_matte[sk_gray!=0]=sk_rgb[sk_gray!=0]
        # cv2.imshow('sk_matte', sk_matte)
        result = S2I.getResult(sk_matte,cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR))
        result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        cv2.imshow('S2I', result)
        created_hair = cv2.addWeighted(original_img, ALPHA, white_img, 1 - ALPHA, 0)
        # created_hair = original_img.copy()
        created_hair[sk_matte!=0] = result[sk_matte!=0]
        cv2.imshow('S2I Area', created_hair)
        # cv2.imshow('absdiff', cv2.absdiff(original_img, result))
        # cv2.imshow('subtract1', cv2.subtract(original_img, result, mask = matte))
        # cv2.imshow('subtract2', cv2.subtract(result, original_img, mask = matte))
        # cv2.imshow('subtract3', cv2.copyTo(result, cv2.cvtColor(cv2.absdiff(original_img, result), cv2.COLOR_BGR2GRAY), cv2.addWeighted(original_img, ALPHA, white_img, 1 - ALPHA, 0)))
        # cv2.imshow('subtract4', cv2.cvtColor(cv2.absdiff(original_img, result), cv2.COLOR_BGR2GRAY))
        mask = cv2.cvtColor(cv2.absdiff(original_img, result), cv2.COLOR_BGR2GRAY)
        # mask = cv2.threshold(img,40,255, cv2.THRESH_TOZERO)
        black_img = np.zeros_like(mask)
        black_img[matte > 0] = mask[matte > 0]
        subtract5 = cv2.addWeighted(original_img, ALPHA, white_img, 1 - ALPHA, 0)
        subtract5[black_img > 0] = result[black_img > 0]
        cv2.imshow('subtract5', subtract5)
        _, ret = cv2.threshold(mask,3,255, cv2.THRESH_TOZERO)
        print(mask.shape)
        print(ret.shape)
        cv2.imshow('ret', ret)
        subtract6 = cv2.addWeighted(original_img, ALPHA, white_img, 1 - ALPHA, 0)
        subtract6[ret > 0] = result[ret > 0]

        cv2.imshow('subtract6', subtract6)
        # created_hair2 = cv2.addWeighted(original_img, 0.85, white_img, 1 - 0.85, 0)
        # for h in range(result.shape[0]):
        #     for w in range(result.shape[1]):
        #         diff = abs(np.sum(result[h,w, :] - original_img[h, w,:]))
        #         if np.sum(sk_matte[h,w,:]) != 0 and diff > 60:
        #             created_hair2[h,w,:] = result[h,w,:]
        # cv2.imshow('created_hair2', created_hair2)
        '''


    
    middle_time = time.time()
    if checkpoint or (middle_time - start_time >= DELAY and drawing):
        start_time = middle_time
        if len(original_backup_list) > 1:
            matte = S2M.getResult(cvtSketchForS2M(sketch_img))
            READY_TO_S2I = True
        else:
            matte = np.zeros_like(sketch_img)
            READY_TO_S2I = False
        # cv2.imshow('matte', matte)
        checkpoint = False
        # print(matte.shape) # (512, 512)
    if READY_TO_S2I and middle_time - start_time >= DELAY:
        start_time = middle_time
        sk_matte = np.array(cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR))
        sk_rgb = cv2.cvtColor(sketch_img,cv2.COLOR_BGR2RGB)
        sk_gray = cv2.cvtColor(sketch_img,cv2.COLOR_BGR2GRAY)
        sk_matte[sk_gray!=0]=sk_rgb[sk_gray!=0]
        cv2.imshow("qewqe", sk_matte)
        result, hair_matte = S2I.getResult(sk_matte,cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR), thred)
        result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        only_hair, _ = S2I.getResult(sk_matte,cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR), 0)
        only_hair = cv2.cvtColor(only_hair,cv2.COLOR_RGB2BGR)
        # matte_thred_img = cv2.addWeighted(matte, matte_thred, np.zeros_like(matte), 1 - matte_thred, 0)
        result[sk_matte!=0] = result[sk_matte!=0]*(1-matte_thred) + cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)[sk_matte!=0] * matte_thred
        # cv2.imshow('S2I', result)
        READY_TO_S2I = False

    before_thred = thred
    before_matte_thred = matte_thred
cv2.destroyAllWindows()
