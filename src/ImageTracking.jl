__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays

include("core.jl")
include("optical_flow.jl")
include("haar.jl")
include("tracker.jl")

export

	# main functions
	optical_flow,
	init_tracker,
	update_tracker,

  # other functions
	haar_coordinates,
	haar_features,
	polynomial_expansion,

	# optical flow algorithms
	LK,
	Farneback,

	# tracking algorithms
	TrackerMedianFlow

end
