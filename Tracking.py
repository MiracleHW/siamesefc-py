import numpy as np
import cv2
import math
from scipy import signal
from SFNet import *

class fc_tracking:

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
    SFnet=SFNet()

    def get_subwindow_tracking(self,im,pos,original_sz,model_sz,av_chans):#pos: [y,x], original_sz:[h,w]
        im_sz=im.shape

        c=((original_sz)+1)/2

        context_xmin = round(pos[1]-c[1])
        context_xmax = context_xmin + original_sz[1]-1
        context_ymin = round(pos[0]-c[0])
        context_ymax = context_ymin + original_sz[0]-1

        left_pad=max(0,-context_xmin)
        top_pad=max(0,-context_ymin)
        right_pad=max(0,context_xmax-im_sz[1])
        bottom_pad=max(0,context_ymax-im_sz[0])

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        left_pad,right_pad,top_pad,bottom_pad=(int)(left_pad),(int)(right_pad),(int)(top_pad),(int)(bottom_pad)
        R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
        R=np.lib.pad(R,((top_pad,bottom_pad),(left_pad,right_pad)),'constant',constant_values=av_chans[0])
        G=np.lib.pad(G,((top_pad,bottom_pad),(left_pad,right_pad)),'constant',constant_values=av_chans[1])
        B=np.lib.pad(B,((top_pad,bottom_pad),(left_pad,right_pad)),'constant',constant_values=av_chans[2])

        context_xmax,context_xmin,context_ymin,context_ymax=self.Int(context_xmax,context_xmin,context_ymin,context_ymax)
        R=R[context_ymin:context_ymax,context_xmin:context_xmax]
        G=G[context_ymin:context_ymax,context_xmin:context_xmax]
        B=B[context_ymin:context_ymax,context_xmin:context_xmax]

        im_patch_original=np.zeros((context_ymax-context_ymin,context_xmax-context_xmin,3))
        #im_patch_original=im[context_ymin:context_ymax,context_xmin:context_xmax,:]
        im_patch_original[:,:,0]=R
        im_patch_original[:,:,1]=G
        im_patch_original[:,:,2]=B

        im_patch=cv2.resize(im_patch_original,((int)(model_sz[0]),(int)(model_sz[1])),interpolation=cv2.INTER_CUBIC)

        return im_patch

    def make_scale_pyramid(self,im,targetPosition,in_side_scale,out_side,avgchan):
        in_side_scale=np.round(in_side_scale)
        max_target_side=in_side_scale[-1]
        min_target_side=in_side_scale[0]
        beta=out_side/min_target_side
        search_side=round(beta*max_target_side)
        search_region=self.get_subwindow_tracking(im,targetPosition,np.array([max_target_side,max_target_side]),[search_side,search_side],avgchan)

        pyramid=np.zeros((self.numScale,out_side,out_side,3))
        for s in range(0,(int)(self.numScale)):
            target_side=round(beta*in_side_scale[s])
            p=self.get_subwindow_tracking(search_region,[(1+search_side)/2,(1+search_side)/2],np.array([target_side,target_side]),[out_side,out_side],avgchan)
            pyramid[s,:,:,:]=p
        return pyramid

    def tracker_eval(self,s_x,x_crops,z_crop,targetPosition,window):

        #reponseMaps=self.SFnet.eval_scoreMap(z_crop,x_crops,e_size=self.exemplarSize,i_size=self.instanceSize)
        reponseMaps = self.SFnet.eval_scoreMap(z_crop, x_crops)
        if self.numScale>1:
            currentScaleID=math.ceil(self.numScale/2)-1
            bestScale=currentScaleID
            bestPeak=-float('inf')
            reponseMapsUp=[]
            for s in range(0,(int)(self.numScale)):
                if self.reponseUp>1.0:
                    map=cv2.resize(reponseMaps[s,:,:,0],(0,0),fx=self.reponseUp,fy=self.reponseUp,interpolation=cv2.INTER_CUBIC)
                reponseMapsUp.append(map)
                thisResponse=map
                if s!=currentScaleID:
                    thisResponse=thisResponse*self.scalePenalty
                thisPeak=np.max(thisResponse)
                if thisPeak>bestPeak:
                    bestPeak=thisPeak
                    bestScale=s
            reponseMap=reponseMapsUp[bestScale]

        reponseMap=reponseMap-np.min(reponseMap)
        reponseMap=reponseMap/np.sum(reponseMap)

        reponseMap=(1-self.wInfluence)*reponseMap+self.wInfluence*window

        r_max,c_max=np.where(reponseMap==np.max(reponseMap))
        r_max=r_max[0]
        c_max=c_max[0]
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

    def combineRGB(self,R,G,B):
        x,y=R.shape
        im=np.zeros((x,y,3))
        im[:,:,0]=R
        im[:,:,1]=G
        im[:,:,2]=B
        return im

    def Init4NumberBB(self):
        bbox=[]
        path='/workspace/hw/WorkSpace/siamese-fc-py/data/BlurBody/groundtruth_rect.txt'
        file=open(path,"r")
        bb=file.readline()
        bb=bb[:-1]
        bb=bb[:-1]
        bb=bb.split("\t",4)
        for b in bb:
            bbox.append((float)(b))
        cx=bbox[0]+bbox[2]/2
        cy=bbox[1]+bbox[3]/2
        return [cx,cy,bbox[2],bbox[3]]

    def Init8NumberBB(self):
        bbox=[]
        path='/workspace/hw/WorkSpace/siamese-fc-py/vot15_bag/groundtruth.txt'
        file=open(path,"r")
        bb=file.readline()
        bb=bb[:-1]
        bb=bb.split(',',8)
        for b in bb:
            bbox.append((float)(b))
        cx=(bbox[0]+bbox[2]+bbox[4]+bbox[6])/4.0
        cy=(bbox[1]+bbox[3]+bbox[5]+bbox[7])/4.0
        x1=min(bbox[0],bbox[2],bbox[4],bbox[6])
        x2=max(bbox[0],bbox[2],bbox[4],bbox[6])
        y1=min(bbox[1],bbox[3],bbox[5],bbox[7])
        y2=max(bbox[1],bbox[3],bbox[5],bbox[7])
        A1=math.sqrt((bbox[0]-bbox[2])**2+(bbox[1]-bbox[3])**2)*math.sqrt((bbox[2]-bbox[4])**2+(bbox[3]-bbox[5])**2)
        A2=(x2-x1)*(y2-y1)
        s=math.sqrt(A1/A2)
        w=s*(x2-x1)+1
        h=s*(y2-y1)+1
        return [cx,cy,w,h]

    def Int(self,*args):
        tup=()
        for x in args:
            x=(int)(x)
            tup=tup+(x,)
        return tup

    def ReadSequences(self):
        imgs = []
        for i in range(1, 196):
            img_path = '/workspace/hw/WorkSpace/siamese-fc-py/vot15_bag/imgs/%08d.jpg' % i
            #img_path = '/workspace/hw/WorkSpace/siamese-fc-py/data/BlurBody/img/%04d.jpg' % i
            img = self.cvreadRGBimg(img_path)
            imgs.append(img)
        return imgs

    def tracker(self):
        inteval=1
        #read imgs and groudth from local file
        imgFiles=self.ReadSequences()
        bb=self.Init8NumberBB()#cx,cy,w,h

        targetPosition=np.array([bb[1],bb[0]])#center_y,center_x
        targetSize=np.array([bb[3],bb[2]])#h,w

        #get the first img
        im=imgFiles[self.startFrame]
        nImgs=len(imgFiles)

        if(im.ndim==2):
            im=self.combineRGB(im,im,im)

        #eval the average value for RGB as padding value
        avgchans=np.array([np.mean(im[:,:,0]),np.mean(im[:,:,1]),np.mean(im[:,:,2])])

        #extent region to add context
        wc_z = targetSize[1] + self.contextAmount * np.sum(targetSize)
        hc_z = targetSize[0] + self.contextAmount * np.sum(targetSize)
        s_z = np.sqrt((float)(wc_z * hc_z))
        #final exemplar region is s_z*s_z
        scale_z = (float)(self.exemplarSize) / (float)(s_z)

        z_crop=self.get_subwindow_tracking(im,targetPosition,np.array([round(s_z),round(s_z)]),model_sz=[self.exemplarSize,self.exemplarSize],av_chans=avgchans)

        han = np.reshape(np.hanning(self.scoreSize * self.reponseUp), (self.scoreSize * self.reponseUp, 1))
        window = np.dot(han,np.transpose(han))
        window = window / np.sum(window)

        #eval search region
        d_search = (self.instanceSize - self.exemplarSize) / 2
        pad = d_search / scale_z
        #final search region is s_x*s_x
        s_x = s_z + 2 * pad
        min_s_x=0.2*s_x
        max_s_x=5.0*s_x

        scales=self.scaleStep**np.arange(math.ceil(self.numScale/2-self.numScale),math.floor(self.numScale/2)+1,1)

        cv2.namedWindow("tracking")
        #define a nparray to save result bbox
        rbboxs=np.zeros((4,nImgs+1))
        for i in range(self.startFrame,nImgs):
            if i>self.startFrame:
                im=imgFiles[i]
                if (im.ndim == 2):
                    im = self.combineRGB(im, im, im)
                scaledInstance=s_x*scales
                scaledTarget=[]
                for s in range(0,(int)(self.numScale)):
                    scaledTarget.append(targetSize*scales[s])

                x_crop=self.make_scale_pyramid(im,targetPosition,scaledInstance,self.instanceSize,avgchans)

                newPosition,newScale=self.tracker_eval(round(s_x),x_crop,z_crop,targetPosition,window)

                targetPosition=newPosition
                s_x=max(min_s_x,min(max_s_x,(1-self.scaleLR)*s_x+self.scaleLR*scaledInstance[newScale]))
                targetSize=(1-self.scaleLR)*targetSize+self.scaleLR*scaledTarget[newScale]

            rectPosition=[targetPosition-targetSize/2,targetSize]
            cv2.rectangle(im,((int)(rectPosition[0][1]),(int)(rectPosition[0][0])),((int)(rectPosition[0][1]+rectPosition[1][1]),(int)(rectPosition[0][0]+rectPosition[1][0])),color=(255,0,0))
            cv2.imshow('tracking',im)
            rbboxs[:,i]=np.array([rectPosition[0][1],rectPosition[0][0],rectPosition[1][1],rectPosition[1][0]])
            print "finish frame NO.",i,";bbox:",rbboxs[:,i]
            c = cv2.waitKey(inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break

        cv2.destroyAllWindows()
        print 'save resutls'
        np.save('./results.npy',rbboxs)

if __name__=='__main__':
    tracker=fc_tracking()
    tracker.tracker()
