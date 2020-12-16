from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
from tqdm.notebook import tqdm
import skimage.exposure


video_dir = "/home/paradeisios/Documents/GITLAB/luminance/example/test.mp4"
work_dir = "/home/paradeisios/Documents/GITLAB/luminance/example/"


class Luminance():
    
    def __init__(self,video,method="linear"):
        self.video = video
        self.frame_count = self.get_frames()
        self.width = int(cv2.VideoCapture(self.video).get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cv2.VideoCapture(self.video).get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    
    def frame_partion(self, work_dir):

        ''' Partition the video into frames and save jpg frames into work_dir

            Inputs : workdir(str) --- Path to the output folder 
            Ouput  : ---- '''

        vidcap = cv2.VideoCapture(self.video)
        success,image = vidcap.read()
        pbar = tqdm(total=self.frame_count)
        count = 0
        while success:
            filename = os.path.join(work_dir,"frame_{}.jpg".format(str(count).zfill(3)))
            cv2.imwrite(filename, image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += 1
            pbar.update(1)
        pbar.close()
   
    def calculate_global_luminance(self,work_dir,save_txt = False):

        ''' Calculate the mean grayscale luminance based on the method specified

            Inputs : workdir(str) --- Path to the output folder 
                     save_txt(bool) --- Yes -> saves dataframe with global luminance scores into work_dir

            Ouput  : global_luminance_array(pd.DataFrame) --- A df with the global luminance scores'''
        
        pbar = tqdm(total=self.frame_count)
        global_luminance_array = np.zeros(self.frame_count)
        frames = [frame for frame in os.listdir(work_dir) if frame.endswith("jpg")]
        for index,frame in enumerate(sorted(frames,key=lambda x: int(x[6:9]))):          
            global_luminance_array[index] = Luminance.frame_luminance(os.path.join(work_dir,frame))
            pbar.update(1)
        pbar.close()
        
        if save_txt:
            filename = os.path.join(work_dir,"global_luminance.txt")
            with open(filename,"w+") as file:
                np.savetxt(file,global_luminance_array)
        
        return pd.DataFrame(data=global_luminance_array, 
                            index=None, 
                            columns=["global"], 
                            dtype=np.float64) 

            
    def local_partition(self,work_dir,pupil_x,pupil_y,r):
    
        ''' Extract portions of frames where subject was looking

        Inputs : workdir(str) --- Path to the output folder 
                 pupil_x      --- Location of pupil on the x-axis
                 pupil_y      --- Location of pupil on the y-axis
                 r            --- Radius of the circle mask
        Ouput  :             -----                                 '''

        
        pbar = tqdm(total=self.frame_count)
        frames = [frame for frame in os.listdir(work_dir) if frame.endswith("jpg")]
        
        for index,frame in enumerate(sorted(frames,key=lambda x: int(x[-7:-4]))):
            
            img = cv2.imread(os.path.join(work_dir,frame),cv2.IMREAD_COLOR)
            mask = np.zeros((self.height,self.width), np.uint8)
            mask = cv2.circle(mask,(pupil_x[index],pupil_y[index]),r,1,thickness=-1)
            
            masked_data = cv2.bitwise_and(img, img, mask=mask)
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
            result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            result[:,:,3] = mask
            filename = os.path.join(work_dir,"pupil_frame_{}.png".format(str(index).zfill(3)))
            cv2.imwrite(filename, result)
            pbar.update(1)
        pbar.close()
        
    def calculate_local_luminance(self,work_dir,save_txt=False):

        ''' Calculate the mean local grayscale luminance based on the method specified

         Inputs : workdir(str) --- Path to the output folder 
                  save_txt(bool) --- Yes -> saves dataframe with global luminance scores into work_dir

         Ouput  : local_luminance_array(pd.DataFrame) --- A df with the global luminance scores'''

        pbar = tqdm(total=self.frame_count)
        local_luminance_array = np.zeros(self.frame_count)
        frames = [frame for frame in os.listdir(work_dir) if frame.startswith("pupil")]
        
        for index,frame in enumerate(sorted(frames,key=lambda x: int(x[-7:-4]))):
            local_luminance_array[index]=Luminance.frame_luminance(frame)
            pbar.update(1)
        pbar.close()
        
        if save_txt:
            filename = os.path.join(work_dir,"local_luminance.txt")
            with open(filename,"w+") as file:
                np.savetxt(file,local_luminance_array)
        
        return pd.DataFrame(data=local_luminance_array, 
                            index=None, 
                            columns=["local"], 
                            dtype=np.float64) 
  
        
    @classmethod
    def frame_luminance(cls,image):
         ''' Calculate the mean grayscale luminance based on the method specified

         Inputs : image(np.array) --- Image to calculate mean luminance
         Ouput  : uminance (float) --- mean luminance score for the image based on the specified method '''
        
         image = np.array(Image.open(image).convert('RGB'))
         linear = lambda x: 0.2126*x[:,:,0]+0.7152*x[:,:,0]+0.0722*x[:,:,2]
         return(np.mean(linear(image)))      
            
            
