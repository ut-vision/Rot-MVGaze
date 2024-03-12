## Dataset

### XGaze HDF dataset

| key | shape | description | 
| -- | -- | -- |
| cam_index | | |
| face_gaze | | |
| face_head_pose | | | 
| face_mat_norm | | | 
| face_patch | | |
| frame_index | | |

### XGaze Camera calibration 

| key | shape | description | 
| -- | -- | -- |
| image_Width | | |
| image_Height | | |
| Camera_Matrix | | | 
| Distortion_Coefficients | | | 
| cam_translation | | |
| cam_rotation | | |

### XGaze Raw annotations
subject****.csv

| columns | description |
| -- | -- |
| 0 | frame id |
| 1 | image name |
| 13 - 149 | detected 2D facial landmarks in original images |
