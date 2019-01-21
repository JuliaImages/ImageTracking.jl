__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays
using LinearAlgebra
using CoordinateTransformations

include("core.jl")
include("optical_flow.jl")
include("haar.jl")
include("utility.jl")

export

	# main functions
    optical_flow,
    optical_flow!,

	# other functions
	haar_coordinates,
	haar_features,

	# other functions
	polynomial_expansion,
	visualize_optical_flow_hsv,

	# optical flow algorithms
	LucasKanade,
	Farneback,

    # types that select implementation
    ConvolutionImplementation,
    MatrixImplementation



end
