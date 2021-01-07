from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm



class Luminance():
    
    METHODS = { "linear" :    lambda x: 0.2126*x[:,:,0]+0.7152*x[:,:,1]+0.0722*x[:,:,2],
                "perceived" : lambda x: 0.299*x[:,:,0]+0.587*x[:,:,1]+0.114*x[:,:,2],
                "average" :   lambda x: np.mean(x,2) }
    
    def __init__(self,video,method="linear"):
        self.video = video
        self.width = int(cv2.VideoCapture(self.video).get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cv2.VideoCapture(self.video).get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.method = Luminance.METHODS[method]
        self.frame_count = self.get_frames()
        
        self.frames = self.frame_partion()

    
    def get_frames(self):

        ''' Precise estimation of frames

            Inputs : ----
            Ouput  : count (int) --- Number of frames in the video '''

        vidcap = cv2.VideoCapture(self.video)
        success,_ = vidcap.read()
        count = 0
        while success:
            success,_ = vidcap.read()
            count+=1
        vidcap.release() 
        cv2.destroyAllWindows()
        return count 
    
    def frame_partion(self):

        ''' Partition the video into frames and save jpg frames into work_dir

            Inputs : -----
            Ouput  : a X by Y by 3 by frames matrix with each frame '''
        partition = np.zeros((self.height,self.width,3,self.frame_count),dtype=np.uint8)
        vidcap = cv2.VideoCapture(self.video)
        success,image = vidcap.read()
        pbar = tqdm(total=self.frame_count)
        count = 0
        
        while success:
            partition[:,:,:,count]=image
            success,image = vidcap.read()
            count += 1
            pbar.update(1)
        pbar.close()
        
        return partition 
    
    def frame_luminance(self,image):
         ''' Calculate the mean grayscale luminance based on the method specified

         Inputs : image(np.array) --- Image to calculate mean luminance
         Ouput  : luminance (float) --- mean luminance score for the image based on the specified method '''
        
         image = np.array(Image.fromarray(image.astype(np.uint8)).convert('RGB')).astype("float64")
         image[image==0]=np.nan
         return(np.nanmean(self.method(image))) 
     
    def get_vid_seconds(self):
        
        ''' Calculate the length of the video in seconds 
        
        Inputs : ----
        Output : seconds(int) '''
        
        vidcap = cv2.VideoCapture(self.video)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(float(totalNoFrames) / float(fps))
    

class Global_Luminance(Luminance):
    
    def __init__(self,video,method="linear"):
        super().__init__(video,method="linear")
        
    def calculate(self,save_txt = False,downsample=False):
        
        ''' Method to estimate per frame luminance 
        
        Inputs : video(str) --- path to video
                 method (See METHODS --- Mathematical model to use for luminance estimation)
                 save_txt(bool) --- Save output in txt file
                 downsample(bool) --- Average across video length
                 
        Output : luminance_array(pd.DataFrame) ---- Luminance array '''
        
        pbar = tqdm(total=self.frame_count)
        global_luminance_array = np.zeros(self.frame_count)
        
        for ii in range(self.frames.shape[3]):
             global_luminance_array[ii] = self.frame_luminance(self.frames[:,:,:,ii])
             pbar.update(1)
        pbar.close()
        
        if downsample:
            seconds = self.get_vid_seconds()
            downsampled = np.array_split(global_luminance_array, seconds)
            global_luminance_array = list(map(np.mean,downsampled))
            
        if save_txt:
            with open("global_luminance.txt","w+") as file:
                np.savetxt(file,global_luminance_array)
        
        return pd.DataFrame(data=global_luminance_array, 
                            index=None, 
                            columns=["global"], 
                            dtype=np.float64) 


class Local_Luminance(Luminance):
    
    def __init__(self,video,pupil_data,radius=30,method="linear"):
        
        super().__init__(video,method="linear")
        self.pupil_x = pupil_data[:,0]
        self.pupil_y = pupil_data[:,1]
        self.radius = radius
        self.local_frames = self.local_partition()
    
    def local_partition(self):
        
        ''' Method to estimate local parts of pupil attendance
        
            Inputs : ----
            
            Outputs: local_frames(np.array) --- a X by Y by 3 by frames matrix with each frame'''
        
        pbar = tqdm(total=self.frame_count)
        local_frames = np.zeros((self.height,self.width,3,self.frame_count),dtype=np.uint8)
        
        for ii in range(self.frames.shape[3]):
            image = self.frames[:,:,:,ii]
            
            mask = np.zeros((self.height,self.width), np.uint8)
            mask = cv2.circle(mask,(self.pupil_x[ii],self.pupil_y[ii]),self.radius,1,thickness=-1)
            
            result = cv2.bitwise_and(image, image, mask=mask)
            
            local_frames[:,:,:,ii]=result
            pbar.update(1)
        pbar.close()
        
        
        return local_frames
    
    
    def calculate(self,save_txt=False,downsample=False):
        
        ''' Method to estimate per frame luminance 
        
        Inputs : video(str) --- path to video
                 method (See METHODS --- Mathematical model to use for luminance estimation)
                 save_txt(bool) --- Save output in txt file
                 downsample(bool) --- Average across video length
                 
        Output : luminance_array(pd.DataFrame) ---- Luminance array '''
        pbar = tqdm(total=self.frame_count)
        local_luminance_array = np.zeros(self.frame_count)
        
        for ii in range(self.frames.shape[3]):
             
             local_luminance_array[ii] = self.frame_luminance(self.local_frames[:,:,:,ii])
             pbar.update(1)
        pbar.close()
        
        if downsample:
            seconds = self.get_vid_seconds()
            downsampled = np.array_split(local_luminance_array, seconds)
            local_luminance_array = list(map(np.mean,downsampled))
    
        if save_txt:
            with open("local_luminance.txt","w+") as file:
                np.savetxt(file,local_luminance_array)
        
        return pd.DataFrame(data=local_luminance_array, 
                            index=None, 
                            columns=["global"], 
                            dtype=np.float64) 
