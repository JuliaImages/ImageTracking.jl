__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations

include("core.jl")
include("optical_flow.jl")

export

	# types
	Coordinate,

	# main functions
	optical_flow,

	# optical flow algorithms
	LK

end
