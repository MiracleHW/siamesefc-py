#!/usr/bin/python2
import cv2
import numpy as np

class FramesInit:
    scales=[]
    search_region=255

    def __init__(self,step=3,rat=0.2):
        for i in range(-step,step+1):
            self.scales.append((1+rat)**(float)(i))

    def ReadSequences(self):
        imgs = []
        for i in range(1, 316):
            img_path = '/disk2/huwei/WorkSpace/data/BlurBody/img/%04d.jpg' % i
            img = cv2.imread(img_path)
            imgs.append(img)
        return imgs

    def ScaleFrame(self,img,bb):
        h, w = img.shape[:2]
        frames=[]
        bboxs=[]
        for scale in self.scales:
            frame=cv2.resize(img,(int(scale*w),int(scale*h)),interpolation=cv2.INTER_CUBIC)
            nbb=[bb[0]*scale,bb[1]*scale,bb[2]*scale,bb[3]*scale]
            frame,bbox=self.FrameRegion(frame,nbb)
            frames.append(frame)
            bboxs.append(bbox)
        return np.array(frames),bboxs

    def FrameRegion(self,frame,last_bb):
        imh, imw = frame.shape[:2]
        x,y,w,h=last_bb
        tpadd,bpadd,lpadd,rpadd=0,0,0,0

        BLACK = [0, 0, 0]
        center_point=[(int)(x)+(int)(w/2),(int)(y)+(int)(h/2)]
        region=[center_point[0]-(int)(self.search_region/2),center_point[1]-(int)(self.search_region/2),self.search_region,self.search_region]

        if region[1]<0:
            tpadd=-region[1]
            region[1]=0
            region[3]=region[3]-tpadd
        if (region[1]+region[3])>imh:
            bpadd=region[1]+region[3]-imh
        if region[0]<0:
            lpadd=-region[0]
            region[0]=0
            region[2]=region[2]-tpadd
        if (region[0]+region[2])>imw:
            rpadd=region[0]+region[2]-imw

        frame=frame[region[1]:region[1]+region[3],region[0]:region[0]+region[2],:]
        frame = cv2.copyMakeBorder(frame, top=(int)(tpadd), bottom=(int)(bpadd), left=(int)(lpadd), right=(int)(rpadd), borderType=cv2.BORDER_CONSTANT, value=BLACK)

        bbox=[122-(int)(w/2),122-(int)(h/2),(int)(w),(int)(h)]

        return frame,bbox

    def InitBB(self):
        bbox=[]
        path='/disk2/huwei/WorkSpace/data/BlurBody/groundtruth_rect.txt'
        file=open(path,"r")
        bb=file.readline()
        bb=bb[:-1]
        bb=bb[:-1]
        bb=bb.split("\t",4)
        for b in bb:
            bbox.append((int)(b))
        return bbox

'''
img_path = '/disk2/huwei/WorkSpace/data/BlurBody/img/%04d.jpg' % 1
img = cv2.imread(img_path)
frameinit=FramesInit()

bbox=frameinit.InitBB()

frames,bbs =frameinit.ScaleFrame(img,bbox)
for i in range(7):
    cv2.rectangle(frames[i],(bbs[i][0],bbs[i][1]),(bbs[i][0] + bbs[i][2], bbs[i][1] + bbs[i][3]), (0, 255, 255), 1)
    cv2.imshow('im',frames[i])
    cv2.waitKey(1000)
'''

