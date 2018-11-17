abstract type TrackerStateEstimator end

#-------------------------------------------------
# STATE ESTIMATOR ALGORITHM FOR BOOSTING TRACKER
#-------------------------------------------------

include("boosting_tracker/online_boosting.jl")

mutable struct TSE_Adaboost{I <: Int, B <: Bool} <: TrackerStateEstimator
    #Initialized
    num_of_base_classifiers::I
    initial_iterations::I
    num_of_features::I
    is_trained::B
    init_patch_size::MVector{2, I}
    sample_ROI::MVector{4, I}

    #Uninitialized
    boost_classifier::StrongClassifier
    replaced_classifier::MVector{N, Int} where N
    swapped_classifier::MVector{N, Int} where N
    current_confidence_map::ConfidenceMap

    TSE_Adaboost(num_of_base_classifiers::I, initial_iterations::I, num_of_features::I, is_trained::B, init_patch_size::MVector{2, I}, sample_ROI::MVector{4, I}) where {I <: Int, B <: Bool} = new{I, B}(
                 num_of_base_classifiers, initial_iterations, num_of_features, is_trained, init_patch_size, sample_ROI)
end

function estimate(estimator::TSE_Adaboost{}, confidence_maps::SVector{N, ConfidenceMap}) where N
    data_type = typeof(estimator.current_confidence_map.states[].responses[])
    images = Array{Array{data_type, 2}, 1}()

    for i = 1:length(estimator.current_confidence_map.states)
        push!(images, estimator.current_confidence_map.states[i].responses)
    end

    index, confidence = classify(estimator.boost_classifier, MVector{length(images)}(images), estimator.sample_ROI)
    return estimator.current_confidence_map.states[index]
end

function update(estimator::TSE_Adaboost{}, confidence_maps::SVector{N, ConfidenceMap}) where N
    if !estimator.is_trained
        num_weak_classifier = 10*estimator.num_of_base_classifiers

        estimator.boost_classifier = initialize_strong_classifier(estimator.num_of_base_classifiers, num_weak_classifier, estimator.initial_iterations, estimator.init_patch_size, true, estimator.sample_ROI)
        init_base_classifiers(estimator.boost_classifier)

        estimator.is_trained = true
    end

    last_confidence_map = confidence_maps[end]
    feature_exchange = estimator.boost_classifier.feature_exchange

    estimator.replaced_classifier =  MVector{length(last_confidence_map.states), Int}()
    estimator.swapped_classifier =  MVector{length(last_confidence_map.states), Int}()

    for  i = 1:ceil(Int, length(last_confidence_map.states)/2)
        current_target_state = last_confidence_map.states[i]

        current_foreground = 1
        if !current_target_state.is_target
            current_foreground = -1
        end

        res = current_target_state.responses

        update(estimator.boost_classifier, res, current_foreground)

        if feature_exchange
            estimator.replaced_classifier[i] = estimator.boost_classifier.replaced_classifier
            estimator.swapped_classifier[i] = estimator.boost_classifier.swapped_classifier

            if estimator.boost_classifier.replaced_classifier > 0 && estimator.boost_classifier.swapped_classifier > 0
                replace_weak_classifier(estimator.boost_classifier, estimator.replaced_classifier[i])
            else
                estimator.replaced_classifier[i] = -1
                estimator.swapped_classifier[i] = -1
            end
        end

        map_position = i + round(Int, length(last_confidence_map.states)/2)
        current_target_state_2 = last_confidence_map.states[map_position]

        current_foreground = 1
        if !current_target_state_2.is_target
            current_foreground = -1
        end

        res_2 = current_target_state_2.responses

        update(estimator.boost_classifier, res_2, current_foreground)

        if feature_exchange
            estimator.replaced_classifier[map_position] = estimator.boost_classifier.replaced_classifier
            estimator.swapped_classifier[map_position] = estimator.boost_classifier.swapped_classifier

            if estimator.boost_classifier.replaced_classifier >= 0 && estimator.boost_classifier.swapped_classifier >= 0
                replace_weak_classifier(estimator.boost_classifier, estimator.replaced_classifier[map_position])
            else
                estimator.replaced_classifier[map_position] = -1
                estimator.swapped_classifier[map_position] = -1
            end
        end
    end
end
