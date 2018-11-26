# ImageTracking

Julia package for optical flow and object tracking algorithms. 

The package currently implements optical flow estimation using the following methods:

1. Lucas-Kanade (for sparse optical flow)
2. Farneback (for dense optical flow)

## Usage Examples

The following sections summarise the key API with the aid of concrete examples. 

### Farneback Algorithm

```julia
"""
    Farneback(Args...)    

A method for dense optical flow estimation developed by Gunnar Farneback. It
computes the optical flow for all the points in the frame using the polynomial
representation of the images. The idea of polynomial expansion is to approximate
the neighbourhood of a point in a 2D function with a polynomial. Displacement
fields are estimated from the polynomial coefficients depending on how the
polynomial transforms under translation.

# Options
Various options for the fields of this type are described in more detail
below.

## Choices for `iterations`

Number of iterations the displacement estimation algorithm is run at each point.
If left unspecified a default value of seven iterations is assumed.

## Choices for `estimation_window`

Determines the neighbourhood size over which information will be intergrated
when determining the displacement of a pixel. The total size equals
`2*estimation_window + 1`.

## Choices for `σ_estimation_window`

Standard deviation of a Gaussian weighting filter used to weigh the contribution
of a pixel's neighbourhood when determining the displacement of a pixel.

## Choices for `expansion_window`

Determines the size of the pixel neighbourhood used to find polynomial expansion
for each pixel; larger values mean that the image will be approximated with
smoother surfaces, yielding more robust algorithm and more blurred motion field.
The total size equals `2*expansion_window + 1`.

## Choices for `σ_expansion_window`

Standard deviation of the Gaussian that is used to smooth the image for the purpose
of approximating it with a polynomial expansion.

# References

Farnebäck G. (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Bigun J.,
Gustavsson T. (eds) Image Analysis. SCIA 2003. Lecture Notes in Computer Science, vol 2749. Springer, Berlin,
Heidelberg

Farnebäck, G.: Polynomial Expansion for Orientation and Motion Estimation. PhD thesis, Linköping University,
Sweden, SE-581 83 Linköping, Sweden (2002) Dissertation No 790, ISBN 91-7373-475-6.

"""
```

An example of dense optical flow estimation using the Farneback algorithm.

```julia
using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames

#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("demo/car2.jpg")
img2 = load("demo/car1.jpg")

algorithm = Farneback(50, estimation_window = 11,
                         σ_estimation_window = 9.0,
                         expansion_window = 6,
                         σ_expansion_window = 5.0)
flow = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2), algorithm)

# Convert from (row,column) to (x,y) convention.
map!(x-> SVector(last(x),first(x)), flow, flow)

# Display optical flow as an image, with hue encoding the orientation and
# saturation encoding the relative magnitude.
max_norm = maximum(map(norm,flow))
normalised_flow = map(PolarFromCartesian(),flow / max_norm)
hsv = zeros(HSV{Float32},size(img1))
for i in eachindex(flow)
    hsv[i] = HSV((normalised_flow[i].θ + pi) * 180 / pi, normalised_flow[i].r, 1)
end
# Visualize the optical flow and save it to disk.
imshow(RGB.(hsv))
save("./demo/optical_flow_farneback.jpg", hsv)
```

<div class="row">
  <div class="column">
   <img src="https://github.com/JuliaImages/ImageTracking.jl/blob/master/demo/car_input.gif" width="320" height="240"/>
  </div>
  <div class="column">
    <img src="https://github.com/JuliaImages/ImageTracking.jl/blob/master/demo/optical_flow_farneback.jpg" width="320" height="240"/>
  </div>
</div>

## Lucas-Kanade Algorithm

### Quick start
```julia
"""
    LucasKanade(Args...)

A differential method for optical flow estimation developed by Bruce D. Lucas
and Takeo Kanade. It assumes that the flow is essentially constant in a local
neighbourhood of the pixel under consideration, and solves the basic optical flow
equations for all the pixels in that neighbourhood by the least squares criterion.

# Options
Various options for the fields of this type are described in more detail
below.

## Choices for `iterations`

The termination criteria of the iterative search algorithm, that is, the number of
iterations.

## Choices for `window_size`

Size of the search window at each pyramid level; the total size of the
window used is 2*window_size + 1.

## Choices for `pyramid_levels`

0-based maximal pyramid level number; if set to 0, pyramids are not used
(single level), if set to 1, two levels are used, and so on.

## Choices for `eigenvalue_threshold`

The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of
optical flow equations, divided by number of pixels in a window; if this value
is less than `eigenvalue_threshold`, then a corresponding feature is filtered
out and its flow is not processed (Default value is 10^-6).

## References

B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision,"
DARPA Image Understanding Workshop, pp 121-130, 1981.

J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas-kanade feature tracker description of the
algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.
"""
```

An example using the Lucas-Kanade algorithm to determine the optical flow.

```julia
using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames

#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("demo/table1.jpg")
img2 = load("demo/table2.jpg")

corners = imcorner(img1, method=shi_tomasi)
I = findall(!iszero, corners)
r, c = (getindex.(I, 1), getindex.(I, 2))
points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)


algorithm = LucasKanade(20, window_size = 11,
                            pyramid_levels = 4,
                            eigenvalue_threshold = 0.000001)
flow, indicator = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2),points, algorithm)

# Keep the subset of points that were succesfully tracked and determine
# correspondences.
valid_points = points[indicator]
valid_flow = flow[indicator]
valid_correspondence = map((x,Δx)-> x+Δx, valid_points, valid_flow)

# Convert (row,columns) to (x,y) convention and round to nearest integer.
pts0 = map(x-> round.(Int,(last(x),first(x))), points)
pts1 = map(x-> round.(Int,(last(x),first(x))), valid_points)
pts2 = map(x-> round.(Int,(last(x),first(x))), valid_correspondence)
lines = map((p1, p2) -> (p1,p2), pts1, pts2)

# Visualise the optical flow. Red lines demarcate optical flow on the keypoints
# that were succesfully tracked.
guidict = imshow(img1)
idx2 = annotate!(guidict, AnnotationLines(lines, linewidth=2.0, color=RGB(1,0,0), coord_order="xyxy"))
```




