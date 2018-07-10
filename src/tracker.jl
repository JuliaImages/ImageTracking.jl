abstract type Tracker end

function init_tracker(tracker::Tracker, image::Array{T, 2}, bounding_box::MVector{4, Int}) where T
    @assert bounding_box[1] >= 1 && bounding_box[1] <= size(image)[1]
    @assert bounding_box[2] >= 1 && bounding_box[2] <= size(image)[2]
    @assert bounding_box[3] >= 1 && bounding_box[3] <= size(image)[1]
    @assert bounding_box[4] >= 1 && bounding_box[4] <= size(image)[2]

    @assert bounding_box[1] < bounding_box[3]
    @assert bounding_box[2] < bounding_box[4]

    init_impl(tracker, image, bounding_box)
end

function update_tracker(tracker::Tracker, image::Array{T, 2}) where T
    update_impl(tracker, image)
end

#---------------------
# TRACKER COMPONENTS
#---------------------

include("core.jl")
include("tracker_state_estimator.jl")
include("tracker_model.jl")
include("tracker_sampler.jl")
include("tracker_features.jl")

#------------------
# IMPLEMENTATIONS
#------------------

include("boosting_tracker/boosting_tracker.jl")
