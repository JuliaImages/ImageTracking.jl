abstract type TrackerModel end

function model_update(model::TrackerModel)
    if model.max_length_cmaps_vector != -1 && length(model.confidence_maps) >= model.max_length_cmaps_vector-1
      model.confidence_maps = model.confidence_maps[floor(Int, model.max_length_cmaps_vector/2) + 1:end]
    end

    if model.max_length_cmaps_vector != -1 && length(model.trajectory) >= model.max_length_cmaps_vector-1
        model.trajectory = model.trajectory[floor(Int, model.max_length_cmaps_vector/2) + 1:end]
    end

    push!(model.confidence_maps, model.current_confidence_map)
    update(model.state_estimator, SVector{length(model.confidence_maps)}(model.confidence_maps))
end

function run_state_estimator(model::TrackerModel)
    target_state = estimate(model.state_estimator, SVector{length(model.confidence_maps)}(model.confidence_maps))
    push!(model.trajectory, target_state)
end

#-----------------------------
# MODEL FOR MIL TRACKER
#-----------------------------

mutable struct MILModel{I <: Int, S <: Symbol} <: TrackerModel
    #Initialized
    max_length_cmaps_vector::I
    mode::S
    win_size::MVector{2, I}
    trajectory::Vector{TrackerTargetState}
    confidence_maps::Vector{ConfidenceMap}

    #Uninitialized
    state_estimator::TrackerStateEstimator
    current_confidence_map::ConfidenceMap
    current_sample::MVector{N, Tuple{Array{T, 2}, MVector{2, I}}} where N where T

    MILModel(max_length_cmaps_vector::I, mode::S, win_size::MVector{2, I}, trajectory::Vector{TrackerTargetState}) where {I <: Int, S <: Symbol} = new{I, S}(
             max_length_cmaps_vector, mode, win_size, trajectory, Vector{ConfidenceMap}())
end

function initialize_mil_model(bounding_box::MVector{4, Int}, max_length_cmaps_vector::Int = 10)
    win_size = MVector{2, Int}(round(Int, bounding_box[3] - bounding_box[1] + 1), round(Int, bounding_box[4] - bounding_box[2] + 1))
    init_state = TrackerTargetState(SVector{2, Float64}(Float64(bounding_box[1]), Float64(bounding_box[2])), round(Int, bounding_box[3] - bounding_box[1] + 1), round(Int, bounding_box[4] - bounding_box[2] + 1), true)
    trajectory = Vector{TrackerTargetState}()
    push!(trajectory, init_state)
    return MILModel(10, :positive, win_size, trajectory)
end

function model_estimation(model::MILModel{}, responses::MVector{N, Array{T, 2}}, add_to_model::Bool) where N where T
    if isempty(model.current_sample)
        throw(ArgumentError("The samples in model estimation are empty."))
    end

    confidence_map = ConfidenceMap(Vector{TrackerTargetState}(), Vector{Float64}())
    for i = 1:length(responses)
        for j = 1:size(responses[i])[2]
            if model.mode == :positive || model.mode == :classify
                foreground = true
            else
                foreground = false
            end
            current_state = TrackerTargetState(Float64.(model.current_sample[i][2][:]), model.win_size[1], model.win_size[2], foreground)
            temp_resp = responses[i][:,j]
            current_state.responses = reshape(temp_resp, length(temp_resp), 1)

            push!(confidence_map.states, current_state)
            push!(confidence_map.confidence, 0.0)
        end
    end

    if add_to_model
        model.current_confidence_map = confidence_map
    end

    return confidence_map
end
