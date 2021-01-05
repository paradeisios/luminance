# luminance
Simple script to extract local and global luminance from a video, given pupil location data.


## Usage

```python
import luminance

global_lum = Global_Luminance(video) # create global lum object
global_lum.calculate(save_txt = True,downsample=True) # estimate per second global luminance

local_lum = Local_Luminance(video,pupil_data)
local_lum.calculate(save_txt = True,downsample=True) 
```
