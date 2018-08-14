abstract type TrackerSampler end

function sample(sampler::TrackerSampler, image::Array{T, 2}, ROI::MVector{4, Int}) where T
    sample_impl(sampler, image, ROI)
end

#-----------------------------------
# CURRENT STATE SAMPLER ALGORITHM
# Used by Boosting Tracker
#-----------------------------------

mutable struct TS_Current_Sample{F <: Float64, S <: Symbol} <: TrackerSampler
    overlap::F
    search_factor::F
    mode::S
    sampling_ROI::MVector{4, Int}
    valid_ROI::MVector{4, Int}
    tracked_patch::MVector{4, Int}
end

TS_Current_Sample(overlap::F = 0.99, search_factor::F = 1.8, mode::S = :positive) where {F <: Float64, S <: Symbol} = TS_Current_Sample{F, S}(
                  overlap, search_factor, mode, MVector{4, Int}(), MVector{4, Int}(), MVector{4, Int}())

function sample_impl(sampler::TS_Current_Sample{}, image::Array{T, 2}, ROI::MVector{4, Int}) where T
    sampler.tracked_patch = ROI
    sampler.valid_ROI = MVector{4}(1, 1, size(image)[1], size(image)[2])

    height = ROI[3] - ROI[1] + 1
    width = ROI[4] - ROI[2] + 1

    tracking_ROI = MVector{4, Int}()
    tracking_ROI[1] = max(0, floor(Int, ROI[1] - height*((sampler.search_factor - 1)/2)) + 1)
    tracking_ROI[2] = max(0, floor(Int, ROI[2] - width*((sampler.search_factor - 1)/2)) + 1)
    tracking_ROI_height = min(floor(Int, height*sampler.search_factor), size(image)[1] - tracking_ROI[1] + 1)
    tracking_ROI_width = min(floor(Int, width*sampler.search_factor), size(image)[2] - tracking_ROI[2] + 1)

    tracking_ROI[3] = tracking_ROI[1] + tracking_ROI_height - 1
    tracking_ROI[4] = tracking_ROI[2] + tracking_ROI_width - 1

    return sampling(sampler, image, tracking_ROI)
end

function sampling(sampler::TS_Current_Sample{}, image::Array{T, 2}, tracking_ROI::MVector{4, Int}) where T
    sample = Array{Tuple{Array{T, 2}, MVector{2, Int}}, 1}()

    if sampler.valid_ROI == tracking_ROI
        sampler.sampling_ROI = tracking_ROI
    else
        sampler.sampling_ROI[1] = (tracking_ROI[1] < sampler.valid_ROI[1]) ? sampler.valid_ROI[1] : tracking_ROI[1]
        sampler.sampling_ROI[2] = (tracking_ROI[2] < sampler.valid_ROI[2]) ? sampler.valid_ROI[2] : tracking_ROI[2]
        sampler.sampling_ROI[3] = (tracking_ROI[3] > sampler.valid_ROI[3]) ? sampler.valid_ROI[3] : tracking_ROI[3]
        sampler.sampling_ROI[4] = (tracking_ROI[4] > sampler.valid_ROI[4]) ? sampler.valid_ROI[4] : tracking_ROI[4]
    end

    if sampler.mode == :positive
        positive_sample = image[sampler.tracked_patch[1]:sampler.tracked_patch[3], sampler.tracked_patch[2]:sampler.tracked_patch[4]]
        positive_sample_pos = MVector{2}(sampler.tracked_patch[1], sampler.tracked_patch[2])
        for i = 1:4
            push!(sample, (positive_sample, positive_sample_pos))
        end

        return sample
    end

    height = sampler.tracked_patch[3] - sampler.tracked_patch[1] + 1
    width = sampler.tracked_patch[4] - sampler.tracked_patch[2] + 1

    upper_left_block = MVector{4, Int}(sampler.valid_ROI[1], sampler.valid_ROI[2], sampler.valid_ROI[1] + height - 1, sampler.valid_ROI[2] + width - 1)
    upper_right_block = MVector{4, Int}(sampler.valid_ROI[1], sampler.valid_ROI[4] - width + 1, sampler.valid_ROI[1] + height - 1, sampler.valid_ROI[4])
    lower_left_block = MVector{4, Int}(sampler.valid_ROI[3] - height + 1, sampler.valid_ROI[2], sampler.valid_ROI[3], sampler.valid_ROI[2] + width - 1)
    lower_right_block = MVector{4, Int}(sampler.valid_ROI[3] - height + 1, sampler.valid_ROI[4] - width + 1, sampler.valid_ROI[3], sampler.valid_ROI[4])

    if sampler.mode == :negative
        push!(sample, (image[upper_left_block[1]:upper_left_block[3], upper_left_block[2]:upper_left_block[4]], MVector{2}(upper_left_block[1], upper_left_block[2])))
        push!(sample, (image[upper_right_block[1]:upper_right_block[3], upper_right_block[2]:upper_right_block[4]], MVector{2}(upper_right_block[1], upper_right_block[2])))
        push!(sample, (image[lower_left_block[1]:lower_left_block[3], lower_left_block[2]:lower_left_block[4]], MVector{2}(lower_left_block[1], lower_left_block[2])))
        push!(sample, (image[lower_right_block[1]:lower_right_block[3], lower_right_block[2]:lower_right_block[4]], MVector{2}(lower_right_block[1], lower_right_block[2])))

        return sample
    end

    step_row = max(1, floor(Int, (1 - sampler.overlap)*height + 0.5))
    step_col = max(1, floor(Int, (1 - sampler.overlap)*width + 0.5))
    max_row = sampler.sampling_ROI[3] - sampler.sampling_ROI[1] - height
    max_col = sampler.sampling_ROI[4] - sampler.sampling_ROI[2] - width

    for i = 1:step_col:max_col
        for j = 1:step_row:max_row
            push!(sample, (image[j+sampler.sampling_ROI[1]-1:j+sampler.sampling_ROI[1]+height-2, i+sampler.sampling_ROI[2]-1:i+sampler.sampling_ROI[2]+width-2], MVector{2}(j+sampler.sampling_ROI[1]-1, i+sampler.sampling_ROI[2]-1)))
        end
    end

    return sample
end
