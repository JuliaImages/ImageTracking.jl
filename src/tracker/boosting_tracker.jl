#-----------------------------------
# Implementation of boosting tracker
#-----------------------------------

function init_tracker(tracker::TrackerBoosting, boxROI::BoxROI)
    int_image = integral_image(boxROI.img)

    # sampling
    tracker.sampler = CurrentSample(tracker.sampler_overlap, tracker.sampler_search_factor, :positive)
    int_box = BoxROI(int_image, boxROI.bound)
    positive_samples = sample_tracker(tracker.sampler, int_box)

    tracker.sampler.mode = :negative
    negative_samples = sample_tracker(tracker.sampler, int_box)

    if isempty(positive_samples) || isempty(negative_samples)
        error("Couldn't get initial samples")
    end

    # compute harr haar_features

    # model

    # Run model estimation and update for initial iterations
end

function update_tracker(tracker::TrackerBoosting, image::AbstractArray)

end
