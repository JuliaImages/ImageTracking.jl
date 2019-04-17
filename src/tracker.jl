abstract type AbstractTracker end

abstract type AbstractROI end
mutable struct BoxROI{T <: AbstractArray, S <: AbstractArray} <: AbstractROI
    img::T
    bound::S
end

mutable struct TrackerBoosting{I <: Int, F <: Float64, B <: BoxROI} <: AbstractTracker
    # initialized
    boxROI::B
    num_of_classifiers::I
    sampler_overlap::F
    sampler_search_factor::F
    initial_iterations::I
    num_of_features::I

    # constructor
    function TrackerBoosting{I, F, B}(box::B, num_of_classifiers::I = 100, sampler_overlap::F = 0.99,
        sampler_search_factor::F = 1.8, initial_iterations::I = 20,
        num_of_features::I = 1050)where {I <: Int, F <: Float64, B <: BoxROI}
        if size(box.bound, 1) != 4
            error("Invalid bounding box size")
        end

        if box.bound[1] < 1 && box.bound[1] > size(box.img)[1]
            error("Invalid bounding box")
        end
        if box.bound[2] < 1 && box.bound[2] > size(box.img)[2]
            error("Invalid bounding box")
        end
        if box.bound[3] < 1 && box.bound[3] > size(box.img)[1]
            error("Invalid bounding box")
        end
        if box.bound[4] < 1 && box.bound[4] > size(box.img)[2]
            error("Invalid bounding box")
        end

        if box.bound[1] > box.bound[3]
            error("Invalid bounding box")
        end
        if box.bound[2] > box.bound[4]
            error("Invalid bounding box")
        end

        new(box, num_of_classifiers, sampler_overlap, sampler_search_factor, initial_iterations, num_of_features)
    end
end

TrackerBoosting(boxROI::B, num_of_classifiers::I, sampler_overlap::F, sampler_search_factor::F,
initial_iterations::I, num_of_features::I) where {I <: Int, F <: Float64, B <: BoxROI} =
TrackerBoosting{I, F, B}(boxROI, num_of_classifiers, sampler_overlap, sampler_search_factor,
initial_iterations, num_of_features)

#---------------------
# TRACKER COMPONENTS
#---------------------

include("core.jl")
# include("tracker_state_estimator.jl")
# include("tracker_model.jl")
# include("tracker_sampler.jl")
# include("tracker_features.jl")

#------------------
# IMPLEMENTATIONS
#------------------

include("boosting_tracker/boosting_tracker.jl")
