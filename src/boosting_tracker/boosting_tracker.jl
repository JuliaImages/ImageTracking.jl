mutable struct TrackerBoosting{I <: Int, F <: Float64} <: Tracker
    #Initialized
    num_of_classifiers::I
    sampler_overlap::F
    sampler_search_factor::F
    initial_iterations::I
    num_of_features::I

    #Uninitialized
    haar_tracker_features::HaarTrackerFeatures
    sampler::TrackerSampler
    model::TrackerModel

    TrackerBoosting(num_of_classifiers::I = 100, sampler_overlap::F = 0.99, sampler_search_factor::F = 1.8, initial_iterations::I = 20, num_of_features::I = 1050) where {I <: Int, F <: Float64} = new{I, F}(
                    num_of_classifiers, sampler_overlap, sampler_search_factor, initial_iterations, num_of_features)
end

function init_impl(tracker::TrackerBoosting, image::Array{T, 2}, bounding_box::MVector{4, Int}) where T
    int_image = integral_image(Gray.(image))

    #Sampling
    tracker.sampler = TS_Current_Sample(tracker.sampler_overlap, tracker.sampler_search_factor, :positive)
    positive_samples = sample_impl(tracker.sampler, int_image, bounding_box)

    tracker.sampler.mode = :negative
    negative_samples = sample_impl(tracker.sampler, int_image, bounding_box)

    if isempty(positive_samples) || isempty(negative_samples)
        throw(ArgumentError("Could not get initial samples."))
    end

    ROI = tracker.sampler.sampling_ROI

    #Compute Haar Features
    tracker.haar_tracker_features = HaarTrackerFeatures(tracker.num_of_features, MVector{2}((bounding_box[3] - bounding_box[1] + 1), (bounding_box[4] - bounding_box[2] + 1)))
    generate_features(tracker.haar_tracker_features, tracker.num_of_features)

    extraction(tracker.haar_tracker_features, MVector{length(positive_samples)}(positive_samples))
    pos_responses = tracker.haar_tracker_features.responses
    extraction(tracker.haar_tracker_features, MVector{length(negative_samples)}(negative_samples))
    neg_responses = tracker.haar_tracker_features.responses

    #Model
    tracker.model = initialize_boosting_model(bounding_box)
    state_estimator = TSE_Adaboost(tracker.num_of_classifiers, tracker.initial_iterations, tracker.num_of_features, false, MVector{2}((bounding_box[3] - bounding_box[1] + 1), (bounding_box[4] - bounding_box[2] + 1)), ROI)
    tracker.model.state_estimator = state_estimator

    #Run model estimation and update for initial_iterations iterations
    for i = 1:tracker.initial_iterations
        #Compute temp features
        temp_haar_features = HaarTrackerFeatures(length(positive_samples)+length(negative_samples), MVector{2}((bounding_box[3] - bounding_box[1] + 1), (bounding_box[4] - bounding_box[2] + 1)))
        generate_features(temp_haar_features, temp_haar_features.num_of_features)

        #Model estimate
        tracker.model.mode = :negative
        tracker.model.current_sample = MVector{length(negative_samples)}(negative_samples)
        model_estimation(tracker.model, neg_responses, true)

        tracker.model.mode = :positive
        tracker.model.current_sample = MVector{length(positive_samples)}(positive_samples)
        model_estimation(tracker.model, pos_responses, true)

        #Model update
        model_update(tracker.model)

        #Get replaced classifier and change the features
        replaced_classifier = tracker.model.state_estimator.replaced_classifier
        swapped_classifier = tracker.model.state_estimator.swapped_classifier

        for j = 1:length(replaced_classifier)
            if replaced_classifier[j] > 0 && swapped_classifier[j] > 0
                swap_feature(tracker.haar_tracker_features, replaced_classifier[j], swapped_classifier[j])
                swap_feature(tracker.haar_tracker_features, swapped_classifier[j], temp_haar_features.features[j])
            end
        end
    end
end

function update_impl(tracker::TrackerBoosting, image::Array{T, 2}) where T
    int_image = integral_image(Gray.(image))

    #Get the last location
    last_target_state = tracker.model.trajectory[end]
    last_target_bounding_box = MVector{4}(round(Int, last_target_state.position[1]), round(Int, last_target_state.position[2]), round(Int, last_target_state.position[1])+last_target_state.height, round(Int, last_target_state.position[2])+last_target_state.width)

    #Sampling new frame based on last location
    tracker.sampler.mode = :classify
    detection_samples = sample_impl(tracker.sampler, int_image, last_target_bounding_box)
    ROI = tracker.sampler.sampling_ROI

    if isempty(detection_samples)
        throw(ArgumentError("Could not get detection samples."))
    end

    classifiers = get_selected_weak_classifier(tracker.model.state_estimator.boost_classifier)
    extractor = tracker.haar_tracker_features
    responses = extract_selected(extractor, MVector{length(classifiers)}(classifiers), MVector{length(detection_samples)}(detection_samples))

    #Predict new location
    tracker.model.mode = :classify
    tracker.model.current_sample = MVector{length(detection_samples)}(detection_samples)
    tracker.model.state_estimator.current_confidence_map = model_estimation(tracker.model, responses, false)
    tracker.model.state_estimator.sample_ROI = ROI

    run_state_estimator(tracker.model)
    current_state = tracker.model.trajectory[end]
    new_bounding_box = MVector{4}(round(Int, current_state.position[1]), round(Int, current_state.position[2]), round(Int, current_state.position[1])+current_state.height, round(Int, current_state.position[2])+current_state.width)

    #Sampling new frame based on new location
    tracker.sampler.mode = :positive
    positive_samples = sample_impl(tracker.sampler, int_image, new_bounding_box)

    tracker.sampler.mode = :negative
    negative_samples = sample_impl(tracker.sampler, int_image, new_bounding_box)

    if isempty(positive_samples) || isempty(negative_samples)
        throw(ArgumentError("Could not get initial samples."))
    end

    #Extract features
    extraction(tracker.haar_tracker_features, MVector{length(positive_samples)}(positive_samples))
    pos_responses = tracker.haar_tracker_features.responses
    extraction(tracker.haar_tracker_features, MVector{length(negative_samples)}(negative_samples))
    neg_responses = tracker.haar_tracker_features.responses

    #Compute temp features
    temp_haar_features = HaarTrackerFeatures(length(positive_samples)+length(negative_samples), MVector{2}((new_bounding_box[3] - new_bounding_box[1] + 1), (new_bounding_box[4] - new_bounding_box[2] + 1)))
    generate_features(temp_haar_features, temp_haar_features.num_of_features)

    #Model estimate
    tracker.model.mode = :negative
    tracker.model.current_sample = MVector{length(negative_samples)}(negative_samples)
    model_estimation(tracker.model, neg_responses, true)

    tracker.model.mode = :positive
    tracker.model.current_sample = MVector{length(positive_samples)}(positive_samples)
    model_estimation(tracker.model, pos_responses, true)

    #Model update
    model_update(tracker.model)

    #Get replaced classifier and change the features
    replaced_classifier = tracker.model.state_estimator.replaced_classifier
    swapped_classifier = tracker.model.state_estimator.swapped_classifier

    for j = 1:length(replaced_classifier)
        if replaced_classifier[j] > 0 && swapped_classifier[j] > 0
            swap_feature(tracker.haar_tracker_features, replaced_classifier[j], swapped_classifier[j])
            swap_feature(tracker.haar_tracker_features, swapped_classifier[j], temp_haar_features.features[j])
        end
    end

    return new_bounding_box
end
