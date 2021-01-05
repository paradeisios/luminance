from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import skimage.exposure



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

            Inputs : workdir(str) --- Path to the output folder 
            Ouput  : ---- '''
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
         Ouput  : uminance (float) --- mean luminance score for the image based on the specified method '''
        
         image = np.array(Image.fromarray(image).convert('RGB'))
         return(np.mean(self.method(image))) 
     
    def get_vid_seconds(self):
        
        vidcap = cv2.VideoCapture(self.video)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(float(totalNoFrames) / float(fps))
    

class Global_Luminance(Luminance):
    
    def __init__(self,video,method="linear"):
        super().__init__(video,method="linear")
        
    def calculate(self,save_txt = False,downsample=False):
        
        pbar = tqdm(total=self.frame_count)
        global_luminance_array = np.zeros(self.frame_count)
        
        for ii in range(lum.frames.shape[3]):
             global_luminance_array[ii] = self.frame_luminance(lum.frames[:,:,:,ii])
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
    
    def __init__(self,video,pupil_data,radius,method="linear"):
        
        super().__init__(video,method="linear")
        self.pupil_x = pupil_data[:,0]
        self.pupil_y = pupil_data[:,1]
        self.radius = radius
        self.local_frames = self.local_partition()
    
    def local_partition(self):
        
        pbar = tqdm(total=self.frame_count)
        local_frames = np.zeros((self.height,self.width,4,self.frame_count),dtype=np.uint8)
        
        for ii in range(self.frames.shape[3]):
            image = self.frames[:,:,:,ii]
            
            mask = np.zeros((self.height,self.width), np.uint8)
            mask = cv2.circle(mask,(self.pupil_x[ii],self.pupil_y[ii]),self.radius,1,thickness=-1)
            
            masked_data = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
            
            ###### thresholding

            _,thresh = cv2.threshold(gray, 11, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(contours[0])
            
            
            # countour
            contour = np.zeros_like(gray)
            cv2.drawContours(contour, [big_contour], 0, 255, -1)
            
            # blur dilate image
            blur = cv2.GaussianBlur(contour, (5,5), sigmaX=0, sigmaY=0, borderType = cv2.BORDER_DEFAULT)
            
            # stretch so that 255 -> 255 and 127.5 -> 0
            mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
            
            # put mask into alpha channel of input
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            result[:,:,3] = mask
            local_frames[:,:,:,ii]=result
            pbar.update(1)
        pbar.close()
        
        return local_frames
    
    
    def calculate(self,save_txt=False,downsample=False):
        
        pbar = tqdm(total=self.frame_count)
        local_luminance_array = np.zeros(self.frame_count)
        
        for ii in range(lum.frames.shape[3]):
             local_luminance_array[ii] = self.frame_luminance(lum.local_frames[:,:,:,ii])
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

        



video = "/home/paradeisios/Documents/GITLAB/luminance/example/test.mp4"
pupil_data = np.random.randint(low=100,high=200,size=(151,2))

lum = Local_Luminance(video,pupil_data,radius = 20,method="average")
b=lum.calculate()

