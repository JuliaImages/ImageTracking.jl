abstract type TrackerModel end

#-----------------------------
# MODEL FOR MEDIANFLOW TRACKER
#-----------------------------

mutable struct TrackerMedianFlowModel{I <: Int} <: TrackerModel
    image::Array{T, 2} where T
    bounding_box::MVector{4, I}
end
