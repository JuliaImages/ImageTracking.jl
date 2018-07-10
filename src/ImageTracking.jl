__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays

include("optical_flow.jl")
include("tracker.jl")

export

	# main functions
	optical_flow,
	init_tracker,
	update_tracker,

	# optical flow algorithms
	LK,

	#tracking algorithms
	TrackerBoosting

end
