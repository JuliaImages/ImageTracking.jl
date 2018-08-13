mutable struct TrackerMIL{F <: Float64, I <: Int} <: Tracker
    #Initialized
    initial_inner_radius::F
    window_size::F
    initial_max_negatives::I
    tracking_inner_radius::F
    tracking_max_positives::I
    tracking_max_negatives::I
    num_of_features::I

    #Uninitialized
    haar_tracker_features::HaarTrackerFeatures
    sampler::TrackerSampler
    model::TrackerModel

    TrackerMIL(initial_inner_radius::F = 3.0, window_size::F = 25.0,  initial_max_negatives::I = 65, tracking_inner_radius::F = 4.0, tracking_max_positives::I = 100000, tracking_max_negatives::I = 65, num_of_features::I = 250) where {F <: Float64, I <: Int}= new{F, I}(
               initial_inner_radius, window_size, initial_max_negatives, tracking_inner_radius, tracking_max_positives, tracking_max_negatives, num_of_features)
end

function init_impl(tracker::TrackerMIL{}, image::Array{T, 2}, bounding_box::MVector{4, Int}) where T
    srand(1)
    int_image = integral_image(Gray.(image))

    tracker.sampler = TS_Current_Sample_Centered(tracker.initial_inner_radius, tracker.window_size, tracker.initial_max_negatives, tracker.tracking_inner_radius, tracker.tracking_max_positives, tracker.tracking_max_negatives, :init_positive)
    #Positive Sampling
    positive_samples = sample(tracker.sampler, int_image, bounding_box)
    #Negative Sampling
    tracker.sampler.mode = :init_negative
    negative_samples = sample(tracker.sampler, int_image, bounding_box)

    if isempty(positive_samples) || isempty(negative_samples)
        throw(ArgumentError("Could not get initial samples."))
    end

    #Compute Haar Features
    tracker.haar_tracker_features = HaarTrackerFeatures(tracker.num_of_features, MVector{2}((bounding_box[3] - bounding_box[1] + 1), (bounding_box[4] - bounding_box[2] + 1)))
    generate_features(tracker.haar_tracker_features, tracker.num_of_features)

    extraction(tracker.haar_tracker_features, MVector{length(positive_samples)}(positive_samples))
    pos_responses = [tracker.haar_tracker_features.responses]
    extraction(tracker.haar_tracker_features, MVector{length(negative_samples)}(negative_samples))
    neg_responses = [tracker.haar_tracker_features.responses]

    #Model
    tracker.model = initialize_mil_model(bounding_box)
    tracker.model.state_estimator = TSE_MIL(tracker.num_of_features, false)

    #Model estimation and update
    tracker.model.mode = :positive
    tracker.model.current_sample = MVector{length(positive_samples)}(positive_samples)
    model_estimation(tracker.model, MVector{length(pos_responses)}(pos_responses), true)

    tracker.model.mode = :negative
    tracker.model.current_sample = MVector{length(negative_samples)}(negative_samples)
    model_estimation(tracker.model, MVector{length(neg_responses)}(neg_responses), true)

    model_update(tracker.model)
end

function update_impl(tracker::TrackerMIL{}, image::Array{T, 2}) where T
    int_image = integral_image(Gray.(image))

    #Get last location
    last_target_state = tracker.model.trajectory[end]
    #TODO: Make -1 in Boosting as well
    last_target_bounding_box = MVector{4}(round(Int, last_target_state.position[1]), round(Int, last_target_state.position[2]), round(Int, last_target_state.position[1])+last_target_state.height - 1, round(Int, last_target_state.position[2])+last_target_state.width - 1)

    #Sampling new frame based on last location
    tracker.sampler.mode = :detect
    detection_samples = sample(tracker.sampler, int_image, last_target_bounding_box)

    if isempty(detection_samples)
        throw(ArgumentError("Could not get detection samples."))
    end

    #Extract features from new samples
    extraction(tracker.haar_tracker_features, MVector{length(detection_samples)}(detection_samples))
    responses = [tracker.haar_tracker_features.responses]

    #Predict new location
    tracker.model.mode = :classify
    tracker.model.current_sample = MVector{length(detection_samples)}(detection_samples)
    tracker.model.state_estimator.current_confidence_map = model_estimation(tracker.model, MVector{length(responses)}(responses), false)

    run_state_estimator(tracker.model)
    current_state = tracker.model.trajectory[end]
    new_bounding_box = MVector{4}(round(Int, current_state.position[1]), round(Int, current_state.position[2]), round(Int, current_state.position[1])+current_state.height-1, round(Int, current_state.position[2])+current_state.width-1)

    #Sampling new frame based on new location
    tracker.sampler.mode = :init_positive
    positive_samples = sample(tracker.sampler, int_image, new_bounding_box)

    tracker.sampler.mode = :init_negative
    negative_samples = sample(tracker.sampler, int_image, new_bounding_box)

    if isempty(positive_samples) || isempty(negative_samples)
        throw(ArgumentError("Could not get initial samples."))
    end

    #Extract features
    extraction(tracker.haar_tracker_features, MVector{length(positive_samples)}(positive_samples))
    pos_responses = [tracker.haar_tracker_features.responses]
    extraction(tracker.haar_tracker_features, MVector{length(negative_samples)}(negative_samples))
    neg_responses = [tracker.haar_tracker_features.responses]

    #Model estimate
    tracker.model.mode = :negative
    tracker.model.current_sample = MVector{length(negative_samples)}(negative_samples)
    model_estimation(tracker.model, MVector{length(neg_responses)}(neg_responses), true)

    tracker.model.mode = :positive
    tracker.model.current_sample = MVector{length(positive_samples)}(positive_samples)
    model_estimation(tracker.model, MVector{length(pos_responses)}(pos_responses), true)

    #Model update
    model_update(tracker.model)

    return new_bounding_box
end
