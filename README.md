# Luminance
Simple script to extract local and global luminance from a video, given pupil location data.

Returns a .csv file.

## Usage

Extract global and local luminance

optional arguments:
  -h, --help           show this help message and exit

#### Required Arguments:
  --video_path     Path with video.\
  --pupil_path     Path to pupil data.\
  --out_path       Path to store the results.\
  --method         Mathematical model to calculate luminance.\
  --downsample     Return the downsampled luminance.\


Methods to estimate luminance supported are linear,perceived and average.
If downsample is specified to True, the script will group frames across seconds and average them.

Example : 
<<<<<<< HEAD
```python luminance.py --video_path=/path/to/video --pupil_path=/path/to/video --out_path=/path/to/out_dir --method=linear```
=======
```python luminance.py --video_path=/path/to/video --pupil_path=/path/to/pupil --out_path=/path/to/out_dir --method=linear --downsample=True```
>>>>>>> 685b944d413573eec0565396ec68ecff4ae41819
