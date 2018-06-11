__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays

include("optical_flow.jl")

export

	# main functions
	optical_flow,

	# other functions
	polynomial_expansion,

	# optical flow algorithms
	Farneback

end
