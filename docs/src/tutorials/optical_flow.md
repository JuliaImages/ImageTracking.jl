# Optical Flow

Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the relative motion between the 
object and camera. It is `2D vector field` where each vector is a displacement vector showing the movement of points from first frame to 
second.

By estimating optical flow between video frames, one can measure the velocities of objects in the video. In general, moving objects that 
are closer to the camera will display more apparent motion than distant objects that are moving at the same speed. Optical flow 
estimation is used in computer vision to characterize and quantify the motion of objects in a video stream, often for motion-based object 
detection and tracking systems.

The ImageTracking package currently houses the following optical flow algorithms:

`Lucas - Kanade`
`Farneback`

The API for optical flow calculation is as follow:

`optical_flow(prev_image, next_image, optical_flow_algo)`

Depending on which optical flow algorithm is used, different outputs are returned by the function.
