# Lucas - Kanade Sparse Optical Flow

A differential method for optical flow estimation developed by Bruce D. Lucas
and Takeo Kanade. It assumes that the flow is essentially constant in a local
neighbourhood of the pixel under consideration, and solves the basic optical flow
equations for all the pixels in that neighbourhood, by the least squares criterion.

The different arguments are:

 -  prev_points       =  Vector of SVector{2} for which the flow needs to be found
 -  next_points       =  Vector of SVector{2} containing initial estimates of new positions of
                         input features in next image
 -  window_size       =  Size of the search window at each pyramid level; the total size of the
                         window used is 2*window_size + 1
 -  max_level         =  0-based maximal pyramid level number; if set to 0, pyramids are not used
                         (single level), if set to 1, two levels are used, and so on
 -  estimate_flag     =  true -> Use next_points as initial estimate
                         false -> Copy prev_points to next_points and use as estimate
 -  term_condition    =  The termination criteria of the iterative search algorithm i.e the number of iterations
 -  min_eigen_thresh  =  The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of optical
                         flow equations, divided by number of pixels in a window; if this value is less than
                         min_eigen_thresh, then a corresponding feature is filtered out and its flow is not processed
                         (Default value is 10^-6)

## References

B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision,"
DARPA Image Understanding Workshop, pp 121-130, 1981.

J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas kanadefeature tracker description of the
algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.

## Example

In this example we will try to find the optical flow for a few corner points `(Shi Tomasi)` between an image and its shifted image.

First we load the image and create the shifted image by shifting it by `5` pixels in the `y` direction (vertically down) and `3` pixels 
in the `x` direction (horizontally right).

```@example 1
using ImageTracking, TestImages, Images, StaticArrays, OffsetArrays

img1 = Gray{Float64}.(testimage("mandrill"))
img2 = OffsetArray(img1, 5, 3)
```

Now we find corners in the image using the `imcorner` function from `Images.jl` since these corners have distinctive properties and are 
much easier for the flow algorithm to track.

```@example 1
corners = imcorner(img1, method=shi_tomasi)
y, x = findn(corners)
a = map((yi, xi) -> SVector{2}(yi, xi), y, x)
```

We have collected all the corners in the Vector `a`. Next we take `200` of these points and find the flow for them.

```@example 1
pts = rand(a, (200,))

lk = LK(pts, [SVector{2}(0.0,0.0)], 11, 4, false, 20)
flow, status, err = optical_flow(img1, img2, lk)
```

The three output Vectors contain the flow values, status and error for each of the points in `pts`.
