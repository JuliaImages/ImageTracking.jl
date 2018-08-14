# Farneback Dense Optical Flow

A method for dense optical flow estimation developed by Gunnar Farneback. It
computes the optical flow for all the points in the frame using the polynomial
representation of the images. The idea of polynomial expansion is to approximate
the neighbourhood of a point in a 2D function with a polynomial. Displacement
fields are estimated from the polynomial coefficients depending on how the
polynomial transforms under translation.

The different arguments are:

 -  flow_est          =  Array of SVector{2} containing estimate flow values for all points in the frame
 -  iterations        =  Number of iterations the displacement estimation algorithm is run at each
                         point
 -  window_size       =  Size of the search window at each pyramid level; the total size of the
                         window used is 2*window_size + 1
 -  σw                =  Standard deviation of the Gaussian weighting filter
 -  neighbourhood     =  size of the pixel neighbourhood used to find polynomial expansion for each pixel;
                         larger values mean that the image will be approximated with smoother surfaces,
                         yielding more robust algorithm and more blurred motion field
 -  σp                =  standard deviation of the Gaussian that is used to smooth derivatives used as a
                         basis for the polynomial expansion (Applicability)
 -  est_flag          =  true -> Use flow_est as initial estimate
                         false -> Assume zero initial flow values
 -  gauss_flag        =  false -> use box filter
                         true -> use gaussian filter instead of box filter of the same size for optical flow
                         estimation; usually, this option gives more accurate flow than with a box filter,
                         at the cost of lower speed (Default Value)

## References

Farnebäck G. (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Bigun J.,
Gustavsson T. (eds) Image Analysis. SCIA 2003. Lecture Notes in Computer Science, vol 2749. Springer, Berlin,
Heidelberg

Farnebäck, G.: Polynomial Expansion for Orientation and Motion Estimation. PhD thesis, Linköping University,
Sweden, SE-581 83 Linköping, Sweden (2002) Dissertation No 790, ISBN 91-7373-475-6.

## Example

In this example we will try to find the optical flow for all the points between an image and its shifted image.

First we load the image and create the shifted image by shifting it by `5` pixels in the `y` direction (vertically down) and `3` pixels 
in the `x` direction (horizontally right).

```@example 1
using ImageTracking, TestImages, Images, StaticArrays

img1 = Gray{Float64}.(testimage("mandrill"))
img2 = similar(img1)
for i = 6:size(img1)[1]
    for j = 4:size(img1)[2]
        img2[i,j] = img1[i-5,j-3]
    end
end
```

Now we calculate the flow for all the points in the image using the `Farneback` algorithm.

```@example 1
fb = Farneback(rand(SVector{2,Float64},2,2), 7, 39, 6.0, 11, 1.5, false, true)
flow = optical_flow(img1, img2, fb)
```

The output Vector contain the flow values for each of the points in the image.
