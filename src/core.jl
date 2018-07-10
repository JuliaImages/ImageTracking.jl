mutable struct TrackerTargetState
    #Initialized
    position::SVector{2, Float64}
    height::Int
    width::Int
    is_target::Bool

    #Uninitialized
    responses::Array{T, 2} where T

    TrackerTargetState(position::SVector{2, Float64}, height::Int, width::Int, is_target::Bool) = new(
                       position, height, width, is_target)
end

mutable struct ConfidenceMap
    states::Vector{TrackerTargetState}
    confidence::Vector{Float64}
end
