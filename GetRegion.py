import numpy as np
import cv2
import math
from scipy import signal

class GetReigon:

    startFrame=0
    exemplarSize=127
    instanceSize=255
    contextAmount=0.5
    numScale=3.0
    scaleStep=1.0375
    scoreSize=17.0
    reponseUp=16.0
    scalePenalty=0.9745
    wInfluence = 0.176
    totalStride = 8.0
    scaleLR = 0.59

    def get_subwindow_tracking(self,im,pos,original_sz,model_sz,av_chans):#pos: [y,x], original_sz:[h,w]
        im_sz=im.shape

        c=((float)(original_sz)+1)/2

        context_xmin = round(pos[1]-c[1])
        context_xmax = context_xmin + original_sz[1]
        context_ymin = round(pos[0]-c[0])
        context_ymax = context_ymin + original_sz[0]

        left_pad=max(0,-context_xmin)
        top_pad=max(0,-context_ymin)
        right_pad=max(0,context_xmax-im_sz[1])
        bottom_pad=max(0,context_ymax-im_sz[0])

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
        np.lib.pad(R,((top_pad,bottom_pad),(right_pad,left_pad)),'constant',av_chans[0])
        np.lib.pad(G,((top_pad,bottom_pad),(right_pad,left_pad)),'constant',av_chans[1])
        np.lib.pad(B,((top_pad,bottom_pad),(right_pad,left_pad)),'constant',av_chans[2])
        im=self.combine_array(R,G,B)

        im_patch_original=im[context_ymin:context_ymax,context_xmin:context_xmax,:]

        im_patch=cv2.resize(im_patch_original,(model_sz[0],model_sz[1]),interpolation=cv2.INTER_CUBIC)

        return im_patch

    def make_scale_pyramid(self,im,targetPosition,in_side_scale,out_side,avgchan):
        in_side_scale=np.round(in_side_scale)
        max_target_side=in_side_scale[-1]
        min_target_side=in_side_scale[0]
        beta=out_side/min_target_side
        search_side=round(beta*max_target_side)
        search_region=self.get_subwindow_tracking(im,targetPosition,np.array([search_side,search_side]),[max_target_side,max_target_side],avgchan)

        pyramid=[]
        for s in range(0,self.numScale):
            target_side=round(beta*in_side_scale[s])
            p=self.get_subwindow_tracking(search_region,[(1+search_side)/2,(1+search_side)/2],np.array([out_side,out_side]),[target_side,target_side],avgchan)
            pyramid.append(p)
        return np.array(pyramid)

    def tracker_eval(self,s_x,x_crops,z_crop,targetPosition,window):
        reponseMaps=[x_crops,z_crop]

        if self.numScale>1:
            currentScaleID=math.ceil(self.numScale/2)-1
            bestScale=currentScaleID
            bestPeak=-float('inf')
            reponseMapsUp=[]
            for s in range(0,self.numScale):
                if self.reponseUp>1:
                    map=cv2.resize(reponseMaps[:,:,s],self.reponseUp,interpolation=cv2.INTER_CUBIC)
                reponseMapsUp.append(map)
                thisResponse=map
                if s!=currentScaleID:
                    thisResponse=thisResponse*self.scalePenalty
                thisPeak=np.max(thisResponse)
                if thisPeak>bestPeak:
                    bestPeak=thisPeak
                    bestScale=s
            reponseMap=reponseMapsUp[:,:,bestScale]

        reponseMap=reponseMap-np.min(reponseMap)
        reponseMap=reponseMap/np.sum(reponseMap)

        reponseMap=(1-self.wInfluence)*reponseMap+self.wInfluence*window

        r_max,c_max=np.where(reponseMap==np.max(reponseMap))
        r_max=r_max[0]
        c_max=c_max[0]
        r_max,c_max=self.avoid_empty_position(r_max,c_max)
        p_corr=np.array([r_max,c_max])
        disp_instanceFinal=p_corr-math.ceil(self.scoreSize*self.reponseUp/2)
        disp_instanceInput=disp_instanceFinal*self.totalStride/self.reponseUp
        disp_instanceFrame=disp_instanceInput*s_x/self.instanceSize
        newTargetPosition=targetPosition+disp_instanceFrame
        return newTargetPosition,bestScale

    def cvreadRGBimg(self,path):
        img=cv2.imread(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img

    def avoid_empty_position(self,r_max,c_max):
        if ~r_max:
            r_max=math.ceil(self.scoreSize/2)
        if ~c_max:
            c_max=math.ceil(self.scoreSize/2)
        return r_max,c_max

    def combine_array(self,*args):
        a=[]
        for element in args:
            a.append(element)
        return np.array(a)

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

    def ReadSequences(self):
        imgs = []
        for i in range(1, 316):
            img_path = '/disk2/huwei/WorkSpace/data/BlurBody/img/%04d.jpg' % i
            img = self.cvreadRGBimg(img_path)
            imgs.append(img)
        return imgs

    def tracker(self):
        #read imgs and groudth from local file
        imgFiles=self.ReadSequences()
        bb=self.InitBB()#x,y,w,h

        targetPosition=np.array([bb[1]+bb[3]/2,bb[0]+bb[2]/2])#center_y,center_x
        targetSize=np.array([bb[3],bb[2]])#h,w

        #get the first img
        im=imgFiles[self.startFrame]
        nImgs=len(imgFiles)

        if(im.ndim==2):
            im=self.combine_array(im,im,im)

        #eval the average value for RGB as padding value
        avgchans=np.array([np.mean(im[:,:,0]),np.mean(im[:,:,1]),np.mean(im[:,:,2])])

        #extent region to add context
        wc_z = targetSize[1] + self.contextAmount * np.sum(targetSize)
        hc_z = targetSize[0] + self.contextAmount * np.sum(targetSize)
        s_z = np.sqrt((float)(wc_z * hc_z))
        #final exemplar region is s_z*s_z
        scale_z = (float)(self.exemplarSize) / (float)(s_z)

        z_crop=self.get_subwindow_tracking(im,targetPosition,np.array([round(s_z),round(s_z)]),model_sz=[self.exemplarSize,self.exemplarSize],av_chans=avgchans)

        window = np.dot(np.hanning(self.scoreSize * self.reponseUp),np.transpose(np.hanning(self.scoreSize * self.reponseUp)))
        window = window / np.sum(window)

        #eval search region
        d_search = (self.instanceSize - self.exemplarSize) / 2
        pad = d_search / scale_z
        #final search region is s_x*s_x
        s_x = s_z + 2 * pad
        min_s_x=0.2*s_x
        max_s_x=5.0*s_x

        scales=self.scaleStep**np.arange(math.ceil(self.numScale/2-self.numScale),math.floor(self.numScale/2)+1,1)

        for i in range(self.startFrame,nImgs):
            if i>self.startFrame:
                im=imgFiles[i]
                if (im.ndim == 2):
                    im = self.combine_array(im, im, im)
                scaledInstance=s_x*scales
                scaledTarget=[targetSize[0]*scales,targetSize[1]*scales]
                x_crop=self.make_scale_pyramid(im,targetPosition,scaledInstance,self.instanceSize,avgchans)

                newPosition,newScale=self.tracker_eval(round(s_x),x_crop,z_crop,targetPosition,window)

                targetPosition=newPosition
                s_x=max(min_s_x,min(max_s_x,(1-self.scaleLR)*s_x+self.scaleLR*scaledInstance[newScale]))
                targetSize=(1-self.scaleLR)*targetSize+self.scaleLR*np.array([scaledTarget[0,newScale],scaledTarget[1,newScale]])
            rectPosition=[]


a=np.arange(math.ceil(3.0/2-3.0),math.floor(3.0/2)+1,1)

b=np.array([[1.2,2.6,1],[1.0,2.0,4.0]])
c=np.dot(np.hanning(16*17),np.transpose(np.hanning(16*17)))
d=np.array([[1.2,2.6],[1.0,2.0]])
e=signal.correlate2d(b,d)
print e
