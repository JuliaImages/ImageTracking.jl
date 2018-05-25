"""
	OpticalFlowAlgo

An optical flow algorithm with given parameters.
"""

abstract type OpticalFlowAlgo end

"""
	optical_flow(prev_img, next_img, algo)

Returns the flow from `prev_img` to `next_image` for the points provided in the `algo`
type using specified algorithm `algo`.
"""

function optical_flow(prev_img::AbstractArray{T, 2}, next_image::AbstractArray{T,2}, algo::OpticalFlowAlgo) where T <: Gray
	# sanity checks
	@assert size.(indices(prev_img)) == size.(indices(next_image)) "Images must have the same size"

	# dispatch appropriate algorithm
	optflow(prev_img, next_image, algo)
end


#------------------
# IMPLEMENTATIONS
#------------------

include("lucas_kanade.jl")
