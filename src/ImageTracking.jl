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

	# optical flow algorithms
	LK

end
