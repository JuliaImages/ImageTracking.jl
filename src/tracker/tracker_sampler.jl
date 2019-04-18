abstract type AbstractROI end
mutable struct BoxROI{T <: AbstractArray, S <: AbstractArray} <: AbstractROI
    img::T
    bound::S
end

abstract type AbstractTrackerSampler end
mutable struct CurrentSample{F <: Float64, S <: Symbol} <: AbstractTrackerSampler
    overlap::F
    search_factor::F
    mode::S
    sampling_ROI::SVector{4, Integer}
    valid_ROI::SVector{4, Integer}
    tracked_patch::SVector{4, Integer}
    CurrentSample{F, S}(overlap, search_factor, mode) where{F <: Float64, S <: Symbol} = new(overlap, search_factor, mode)
end

CurrentSample(overlap::F = 0.99, search_factor::F = 1.8, mode::S = :positive) where{F <: Float64, S <: Symbol} =
CurrentSample{F, S}(overlap, search_factor, mode)

function sample_tracker(sampler::CurrentSample, box::BoxROI)
    sampler.tracked_patch = box.bound
    sampler.valid_ROI = SVector{4, Integer}([1, 1, size(box.img, 1), size(box.img, 2)])

    height = box.bound[3] - box.bound[1] + 1
    width = box.bound[4] - box.bound[2] + 1

    ROI_min_y = max(0, floor(Integer, box.bound[1] - height*((sampler.search_factor - 1) / 2) + 1))
    ROI_min_x = max(0, floor(Integer, box.bound[2] - width*((sampler.search_factor - 1) / 2) + 1))

    ROI_height = min(floor(Integer, height*sampler.search_factor), size(box.img, 1) - ROI_min_y + 1)
    ROI_width = min(floor(Integer, width*sampler.search_factor), size(box.img, 2) - ROI_min_x + 1)

    ROI_max_y = ROI_min_y + ROI_height - 1
    ROI_max_x = ROI_min_x + ROI_width - 1

    ROI = SVector{4, Integer}([ROI_min_y, ROI_min_x, ROI_max_y, ROI_max_x])

    sample_pos_neg_roi(sampler, box, ROI)
end

function sample_pos_neg_roi(sampler::CurrentSample, box::BoxROI, ROI::SVector{4, Integer})
    println(ROI)
    println(sampler.valid_ROI)
    if sampler.valid_ROI == ROI
        sampler.sampling_ROI = ROI
    else
        sampler.sampling_ROI = SVector{4, Integer}(max(sampler.valid_ROI[1], ROI[1]), max(sampler.valid_ROI[2], ROI[2]),
        min(sampler.valid_ROI[3], ROI[3]), min(sampler.valid_ROI[4], ROI[4]))
    end

    if sampler.mode == :positive
        positive_sample = box.img[sampler.tracked_patch[1]:sampler.tracked_patch[3],
         sampler.tracked_patch[2]:sampler.tracked_patch[4]]
        positive_sample_pos = SVector{2, Integer}(sampler.tracked_patch[1], sampler.tracked_patch[2])
        sample = fill((positive_sample, positive_sample_pos), (1, 4))
        return sample
    end

    height = sampler.tracked_patch[3] - sampler.tracked_patch[1] + 1
    width = sampler.tracked_patch[4] - sampler.tracked_patch[2] + 1

    tl_block = SVector{4, Integer}(sampler.valid_ROI[1], sampler.valid_ROI[2], sampler.valid_ROI[1] + height - 1,
    sampler.valid_ROI[2] + width - 1)
    tr_block = SVector{4, Integer}(sampler.valid_ROI[1], sampler.valid_ROI[4] - width + 1, sampler.valid_ROI[1] + height - 1,
    sampler.valid_ROI[4])
    bl_block = SVector{4, Integer}(sampler.valid_ROI[3] - height + 1, sampler.valid_ROI[2], sampler.valid_ROI[3],
    sampler.valid_ROI[2] + width - 1)
    br_block = SVector{4, Integer}(sampler.valid_ROI[3] - height + 1, sampler.valid_ROI[4] - width + 1, sampler.valid_ROI[3],
    sampler.valid_ROI[4])

    if sampler.mode == :negative
        tl_sample = box.img[tl_block[1]:tl_block[3], tl_block[2]:tl_block[4]]
        tr_sample = box.img[tr_block[1]:tr_block[3], tr_block[2]:tr_block[4]]
        bl_sample = box.img[bl_block[1]:bl_block[3], bl_block[2]:bl_block[4]]
        br_sample = box.img[br_block[1]:br_block[3], br_block[2]:br_block[4]]
        sample = [(tl_sample, SVector{2, Integer}(tl_block[1:2])), (tr_sample, SVector{2, Integer}(tr_block[1:2])),
        (bl_sample, SVector{2, Integer}(bl_block[1:2])), (br_sample, SVector{2, Integer}(br_block[1:2]))]

        return sample
    end

    step_row = max(1, floor(Int, (1 - sampler.overlap)*height + 0.5))
    step_col = max(1, floor(Int, (1 - sampler.overlap)*width + 0.5))
    max_row = sampler.sampling_ROI[3] - sampler.sampling_ROI[1] - height
    max_col = sampler.sampling_ROI[4] - sampler.sampling_ROI[2] - width

    sample = Array{Tuple{AbstractArray, SVector{2, Integer}}}
    for i = 1:step_col:max_col
        for j = 1:step_row:max_row
            push!(sample, (box.img[j + sampler.sampling_ROI[1] - 1:j + sampler.sampling_ROI[1] + height - 2,
            i + sampler.sampling_ROI[2] - 1:i + sampler.sampling_ROI[2] + width - 2],
            SVector{2, Integer}(j + sampler.sampling_ROI[1]-1, i + sampler.sampling_ROI[2]-1)))
        end
    end

    return sample
end
