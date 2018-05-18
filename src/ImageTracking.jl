__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations

include("optical_flow.jl")

export

	# main functions
	optical_flow,

	# optical flow algorithms
	LK

end
