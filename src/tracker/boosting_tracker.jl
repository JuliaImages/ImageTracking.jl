#-----------------------------------
# Implementation of boosting tracker
#-----------------------------------

function initialize!(tracker::TrackerBoosting)
    int_image = integral_image(tracker.box.img)

    # sampling
    int_box = BoxROI(int_image, tracker.box.bound)

    tracker.sampler.mode = :positive
    positive_samples = sample_roi(tracker.sampler, int_box)

    tracker.sampler.mode = :negative
    negative_samples = sample_roi(tracker.sampler, int_box)

    if isempty(positive_samples) || isempty(negative_samples)
        error("Couldn't get initial samples")
    end
    # compute harr haar_features

    # model

    # Run model estimation and update for initial iterations
end

function update!(tracker::TrackerBoosting, image::AbstractArray)

end
