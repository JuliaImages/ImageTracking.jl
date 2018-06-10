__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays

include("haar.jl")

export

	# main functions

	# other functions
	haar_coordinates,
	haar_features

	# optical flow algorithms

end
