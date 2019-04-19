#---------------------
# TRACKER COMPONENTS
#---------------------

include("../core.jl")
include("tracker_sampler.jl")
# include("tracker_state_estimator.jl")
# include("tracker_model.jl")
# include("tracker_features.jl")

abstract type AbstractTracker end
mutable struct TrackerBoosting{I <: Integer, B <: BoxROI, CS <: CurrentSampler} <: AbstractTracker
    # initialized
    box::B
    sampler::CS
    num_of_classifiers::I
    initial_iterations::I
    num_of_features::I

    # constructor
    function TrackerBoosting{I, B, CS}(box::B, sampler::CS, num_of_classifiers::I = 100, initial_iterations::I = 20,
        num_of_features::I = 1050)where {I <: Integer, B <: BoxROI, CS <: CurrentSampler}
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
        new(box, sampler, num_of_classifiers, initial_iterations, num_of_features)
    end
end

TrackerBoosting(box::B, sampler::CS, num_of_classifiers::I, initial_iterations::I, num_of_features::I) where{I <: Integer, B <: BoxROI, CS <: CurrentSampler} = TrackerBoosting{I, B, CS}(box, sampler, num_of_classifiers, initial_iterations, num_of_features)


#------------------
# IMPLEMENTATIONS
#------------------

include("boosting_tracker.jl")
