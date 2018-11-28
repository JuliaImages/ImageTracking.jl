"""
	OpticalFlowAlgorithm

An optical flow algorithm with given parameters.
"""
abstract type OpticalFlowAlgorithm end


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
`2*estimation_window + 1`. If left unspecified a default value of eleven is
assumed.

## Choices for `σ_estimation_window`

Standard deviation of a Gaussian weighting filter used to weigh the contribution
of a pixel's neighbourhood when determining the displacement of a pixel.
If left unspecified a default value of nine is assumed.

## Choices for `expansion_window`

Determines the size of the pixel neighbourhood used to find polynomial expansion
for each pixel; larger values mean that the image will be approximated with
smoother surfaces, yielding more robust algorithm and more blurred motion field.
The total size equals `2*expansion_window + 1`. If left unspecified a default
value of six is assumed.

## Choices for `σ_expansion_window`

Standard deviation of the Gaussian that is used to smooth the image for the purpose
of approximating it with a polynomial expansion. If left unspecified a default
value of five is assumed.

# References

1. Farnebäck G. (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Bigun J., Gustavsson T. (eds) Image Analysis. SCIA 2003. Lecture Notes in Computer Science, vol 2749. Springer, Berlin, Heidelberg
2. Farnebäck, G.: Polynomial Expansion for Orientation and Motion Estimation. PhD thesis, Linköping University, Sweden, SE-581 83 Linköping, Sweden (2002) Dissertation No 790, ISBN 91-7373-475-6.

"""
struct Farneback{F <: Float64, I <: Int} <: OpticalFlowAlgorithm
    iterations::I
    estimation_window::I
    σ_estimation_window::F
    expansion_window::I
    σ_expansion_window::F
end

Farneback(iterations::Int; estimation_window::Int = 11, σ_estimation_window::Real = 9.0, expansion_window::Int =  6, σ_expansion_window::Real = 5.0) = Farneback(iterations, estimation_window, σ_estimation_window, expansion_window, σ_expansion_window)

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
window used is 2*window_size + 1. If left unspecified, a default value of
eleven is assumed.

## Choices for `pyramid_levels`

0-based maximal pyramid level number; if set to 0, pyramids are not used
(single level), if set to 1, two levels are used, and so on. IF left unspecified,
a default value of four is assumed.

## Choices for `eigenvalue_threshold`

The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of
optical flow equations, divided by number of pixels in a window; if this value
is less than `eigenvalue_threshold`, then a corresponding feature is filtered
out and its flow is not processed (default value is 10^-6).

## References

1. B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision," DARPA Image Understanding Workshop, pp 121-130, 1981.
2. J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas-kanade feature tracker description of the algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.
"""
struct LucasKanade{F <: Float64, I <: Int}  <: OpticalFlowAlgorithm
	iterations::I
    window_size::I
    pyramid_levels::I
    eigenvalue_threshold::F
end

LucasKanade(iterations::Int = 20; window_size::Int = 11, pyramid_levels::Int = 4, eigenvalue_threshold::Real = 0.000001) = LucasKanade(iterations, window_size,  pyramid_levels, eigenvalue_threshold)

"""
	flow = optical_flow(source, target, Farneback(Args...))
	flow = optical_flow(source, target, displacement, Farneback(Args...))

Returns the dense optical flow from the `source` to the `target` image using the `Farneback` algorithm.

# Details

The `source` and `target` images must be `Gray` types.

The `displacement` argument allows you to specify an initial
guess for the optical flow and must be of type `Array{SVector{2, Float64}, 2}`.
The elements of `displacement` should represent the flow required to map the
`(row, column)` of each pixel in the `source` image to the `target` image.

"""
function optical_flow(source::AbstractArray{T, 2}, target::AbstractArray{T,2}, algorithm::Farneback) where T <: Gray
	# Sanity checks.
	@assert size.(axes(source)) == size.(axes(target)) "Images must have the same size"
	optflow(source, target, algorithm)
end

function optical_flow(source::AbstractArray{T, 2}, target::AbstractArray{T,2}, displacement::Array{SVector{2, Float64}, 2}, algorithm::Farneback) where T <: Gray
	# Sanity checks.
	@assert size.(axes(source)) == size.(axes(target)) "Images must have the same size"
	@assert size.(axes(source)) == size.(axes(displacement)) "Optical flow field must match image size"
	optflow!(source, target, copy(displacement), algorithm)
end

"""
	flow, indicator  = optical_flow(source, target, points, LucasKanade(Args...))
	flow, indicator  = optical_flow(source, target, points, displacement, LucasKanade(Args...))

Returns the optical flow from the `source` to the `target` image for the specified `points` using the `LucasKanade` algorithm.

# Details

The `source` and `target` images must be `Gray` types.

The `points` argument is of type `Array{SVector{2, Float64}, 2}` and represents
a set of keypoints in the `source` image for which the optical flow is to be
computed. The coordinates  of a `point` represent the  `(row, column)` of a
pixel in the image. The `displacement` argument allows you to specify an initial
guess for the optical flow.

The function returns `flow` of type `Array{SVector{2, Float64}, 2}` which
matches the length of `points` and represents the displacement needed to map
a point in the `source` image to the `target` image.

The `indicator` is a vector of boolean values (one for each point in `points`)
which signals whether a particular point was successfully tracked or not.

In order to use the `flow` to index the corresponding point in the `target` image
you first need to round it to the nearest integer, and check that it falls within
the bounds of the `target` image dimensions.

"""
function optical_flow(source::AbstractArray{T, 2}, target::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1}, algorithm::LucasKanade) where T <: Gray
	# Sanity checks.
	@assert size.(axes(source)) == size.(axes(target)) "Images must have the same size"

	optflow(source, target, points, algorithm)
end


function optical_flow(source::AbstractArray{T, 2}, target::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1},  displacement::Array{SVector{2, Float64}, 1}, algorithm::LucasKanade) where T <: Gray
	# sanity checks
	@assert size.(axes(source)) == size.(axes(target)) "Images must have the same size"
	@assert size.(axes(points)) == size.(axes(displacement)) "Vector of points and vector of displacement must have the same size"
	optflow!(source, target, points, copy(displacement), algorithm)
end


#------------------
# IMPLEMENTATIONS
#------------------

include("lucas_kanade.jl")
include("farneback.jl")
