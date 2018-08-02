__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays

include("core.jl")
include("optical_flow.jl")

export

	# main functions
	optical_flow,

	# other functions
	polynomial_expansion,

	# optical flow algorithms
	LK,
	Farneback

end
