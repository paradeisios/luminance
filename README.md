# luminance
Simple script to extract local and global luminance from a video, given pupil location data.

## Usage
5 step process 

```python
from luminance import *
lum = Luminance(video) # define a luminance object
lum.frame_partion(ouput_dir) # returns each frame of a video
lum.calculate_global_luminance(ouput_dir) # returns global luminance
lum.local_partition(ouput_dir) # returns each frame of a video masked at the pupil position
lum.alculate_local_luminance(ouput_dir) # returns local luminance based on pupil position
```
