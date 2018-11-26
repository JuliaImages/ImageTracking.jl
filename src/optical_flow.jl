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
struct Farneback{F <: Float64, I <: Int} <: OpticalFlowAlgorithm
    iterations::I
    estimation_window::I
    σ_estimation_window::F
    expansion_window::I
    σ_expansion_window::F
end

Farneback(iterations::Int; estimation_window::Int = 39, σ_estimation_window::Real = 6.0, expansion_window::Int =  11, σ_expansion_window::Real = 1.5) = Farneback(iterations, estimation_window, σ_estimation_window, expansion_window, σ_expansion_window)

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
struct LucasKanade{F <: Float64, I <: Int}  <: OpticalFlowAlgorithm
	iterations::I
    window_size::I
    pyramid_levels::I
    eigenvalue_threshold::F
end

LucasKanade(iterations::Int = 20; window_size::Int = 11, pyramid_levels::Int = 4, eigenvalue_threshold::Real = 0.000001) = LucasKanade(iterations, window_size,  pyramid_levels, eigenvalue_threshold)

"""
	optical_flow(prev_img, next_img, algo)

Returns the flow from `prev_img` to `next_image` for the points provided in the `algo`
type using specified algorithm `algo`.
"""
function optical_flow(prev_img::AbstractArray{T, 2}, next_image::AbstractArray{T,2}, algo::OpticalFlowAlgorithm) where T <: Gray
	# sanity checks
	@assert size.(axes(prev_img)) == size.(axes(next_image)) "Images must have the same size"
	optflow(prev_img, next_image, algo)
end

function optical_flow!(prev_img::AbstractArray{T, 2}, next_image::AbstractArray{T,2}, displacement::Array{SVector{2, Float64}, 2}, algo::Farneback) where T <: Gray
	# sanity checks
	@assert size.(axes(prev_img)) == size.(axes(next_image)) "Images must have the same size"
	@assert size.(axes(prev_img)) == size.(axes(displacement)) "Optical flow field must match image size"
	optflow!(prev_img, next_image, displacement, algo)
end

function optical_flow(prev_img::AbstractArray{T, 2}, next_image::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1}, algo::LucasKanade) where T <: Gray
	# sanity checks
	@assert size.(axes(prev_img)) == size.(axes(next_image)) "Images must have the same size"

	optflow(prev_img, next_image, points, algo)
end

function optical_flow!(prev_img::AbstractArray{T, 2}, next_image::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1},  displacement::Array{SVector{2, Float64}, 1}, algo::LucasKanade) where T <: Gray
	# sanity checks
	@assert size.(axes(prev_img)) == size.(axes(next_image)) "Images must have the same size"
	@assert size.(axes(points)) == size.(axes(displacement)) "Vector of points and vector of displacement must have the same size"
	optflow!(prev_img, next_image, points, displacement, algo)
end


#------------------
# IMPLEMENTATIONS
#------------------

include("lucas_kanade.jl")
include("farneback.jl")
