abstract type TrackerStateEstimator end

#---------------------------------------------
# STATE ESTIMATOR ALGORITHM FOR MIL TRACKER
#---------------------------------------------

include("mil_tracker/online_mil.jl")

mutable struct TSE_MIL{I <: Int, B <: Bool} <: TrackerStateEstimator
    #Initialized
    num_of_features::I
    trained::B

    #Uninitialized
    boost_classifier::MIL_Boost_Classifier
    current_confidence_map::ConfidenceMap

    TSE_MIL(num_of_features::I, trained::B) where {I <: Int, B <: Bool} = new{I, B}(
            num_of_features, trained)
end

function estimate(estimator::TSE_MIL{}, confidence_maps::SVector{N, ConfidenceMap}) where N
    positive_states, negative_states = prepare_data(estimator, estimator.current_confidence_map)
    prob = classify(estimator.boost_classifier, positive_states)
    best_ind = indmax(prob)
    return estimator.current_confidence_map.states[best_ind]
end

function prepare_data(estimator::TSE_MIL{}, confidence_map::ConfidenceMap)
    pos_counter = 0
    neg_counter = 0
    for i = 1:length(confidence_map.states)
        if confidence_map.states[i].is_target
            pos_counter += 1
        else
            neg_counter += 1
        end
    end

    positive_states = Array{Float64}(pos_counter, max(estimator.num_of_features, length(confidence_map.states[1].responses)))
    negative_states = Array{Float64}(neg_counter, max(estimator.num_of_features, length(confidence_map.states[1].responses)))
    pc = 1
    nc = 1
    for i = 1:length(confidence_map.states)
        current_target_state = confidence_map.states[i]
        state_features = current_target_state.responses
        if current_target_state.is_target
            for j = 1:size(state_features)[1]
                positive_states[pc, j] = state_features[j, 1]
            end
            pc += 1
        else
            for j = 1:size(state_features)[1]
                negative_states[nc, j] = state_features[j, 1]
            end
            nc += 1
        end
    end
    return positive_states, negative_states
end

function update(estimator::TSE_MIL{}, confidence_maps::SVector{N, ConfidenceMap}) where N
    if !estimator.trained
        estimator.boost_classifier = initialize_ClfMilBoost()
        estimator.trained = true
    end

    last_confidence_map = confidence_maps[end]
    positive_states, negative_states = prepare_data(estimator, last_confidence_map)
    update(estimator.boost_classifier, positive_states, negative_states)
end
