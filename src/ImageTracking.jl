__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays

include("core.jl")
include("optical_flow.jl")
include("haar.jl")

export

	# main functions
	optical_flow,
	init_tracker,
	update_tracker,

	# other functions
	polynomial_expansion,
	haar_coordinates,
	haar_features,

	# optical flow algorithms
	LK,
	Farneback,

	# tracking algorithms
	TrackerMIL

end
