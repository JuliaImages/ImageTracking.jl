# ImageTracking

Julia package for Optical Flow and Object Tracking Algorithms

## Lucas-Kanade Algorithm

### Quick start
```julia
"""
    LucasKanade(Args...)

A differential method for optical flow estimation developed by Bruce D. Lucas
and Takeo Kanade. It assumes that the flow is essentially constant in a local
neighbourhood of the pixel under consideration, and solves the basic optical flow
equations for all the pixels in that neighbourhood by the least squares criterion.

The different arguments are:

 -  window_size           =  Size of the search window at each pyramid level; the total size of the
                             window used is 2*window_size + 1
 -  max_level             =  0-based maximal pyramid level number; if set to 0, pyramids are not used
                             (single level), if set to 1, two levels are used, and so on

 -  iterations            =  The termination criteria of the iterative search algorithm i.e the number of iterations
 -  eigenvalue_threshold  =  The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of optical
                             flow equations, divided by number of pixels in a window; if this value is less than
                             min_eigen_thresh, then a corresponding feature is filtered out and its flow is not processed
                             (Default value is 10^-6)

## References

B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision,"
DARPA Image Understanding Workshop, pp 121-130, 1981.

J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas-kanade feature tracker description of the
algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.
"""
```

An example using the Lucas-Kanade algorithm to determine the optical flow.

```julia
using Images, TestImages, ImageTracking, ImageView, CoordinateTransformations, Gtk.ShortNames

# Create two images (the second is a translation of the first).
img1 = Gray{Float64}.(testimage("mandrill"))
img2 = warp(img1, Translation(-1.0, -3.0), axes(img1))

# Find keypoints in the first image.
corners = imcorner(img1, method=shi_tomasi)
I = findall(!iszero, corners)
r, c = (getindex.(I, 1), getindex.(I, 2))
points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)

# The indicator array flags whether a point in `points` was successfully tracked or not. 
# For unsuccessful points the flow was set to zero. 
flow, indicator  = optical_flow(img1, img2, points, LucasKanade(11, 4, 20, 0.000001))

valid_points = points[indicator]
valid_flow = flow[indicator]
valid_correspondence = map((x,Δx)-> x+Δx, valid_points, valid_flow)
```

We can visualise the optical flow using the `ImageView` package.

```julia
# Convert (r,c) to (x,y) convention and round to nearest integer.
pts0 = map(x-> round.(Int,(last(x),first(x))), points)
pts1 = map(x-> round.(Int,(last(x),first(x))), valid_points)
pts2 = map(x-> round.(Int,(last(x),first(x))), valid_correspondence)
lines = map((p1, p2) -> (p1,p2), pts1, pts2)

guidict = imshow(img1)
# Green points indicate the initial keypoints. 
idx1 = annotate!(guidict, AnnotationPoints(pts0, shape='.', size=1, color=RGB(0,1,0)))
# Red lines demarcate optical flow on the keypoints that were succesfully tracked.
idx2 = annotate!(guidict, AnnotationLines(lines, linewidth=2.0, color=RGB(1,0,0), coord_order="xyxy"))
```
