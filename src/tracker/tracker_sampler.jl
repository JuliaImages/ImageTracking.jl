abstract type AbstractROI end
mutable struct BoxROI{T <: AbstractArray, S <: AbstractArray} <: AbstractROI
    img::T
    bound::S
end

abstract type AbstractTrackerSampler end
mutable struct CurrentSampler{F <: AbstractFloat} <: AbstractTrackerSampler
    overlap::F
    search_factor::F
    mode::S where S <: Symbol
    sampling_region::SVector{4, Integer}
    valid_region::SVector{4, Integer}
    tracked_patch::SVector{4, Integer}
    CurrentSampler{F}(overlap, search_factor) where{F <: AbstractFloat} = new(overlap, search_factor)
end

CurrentSampler(overlap::F = 0.99, search_factor::F = 1.8) where{F <: AbstractFloat} = CurrentSampler{F}(overlap, search_factor)

function sample_roi(sampler::CurrentSampler, box::BoxROI)
    sampler.tracked_patch = box.bound
    sampler.valid_region = SVector{4, Integer}([1, 1, size(box.img, 1), size(box.img, 2)])

    height = box.bound[3] - box.bound[1] + 1
    width = box.bound[4] - box.bound[2] + 1

    sampling_region_min_y = max(0, floor(Integer, box.bound[1] - height*((sampler.search_factor - 1) / 2) + 1))
    sampling_region_min_x = max(0, floor(Integer, box.bound[2] - width*((sampler.search_factor - 1) / 2) + 1))

    sampling_region_height = min(floor(Integer, height*sampler.search_factor), size(box.img, 1) - sampling_region_min_y + 1)
    sampling_region_width = min(floor(Integer, width*sampler.search_factor), size(box.img, 2) - sampling_region_min_x + 1)

    sampling_region_max_y = sampling_region_min_y + sampling_region_height - 1
    sampling_region_max_x = sampling_region_min_x + sampling_region_width - 1

    sampling_region = SVector{4, Integer}([sampling_region_min_y, sampling_region_min_x, sampling_region_max_y, sampling_region_max_x])

    sample_mode_roi(sampler, box, sampling_region)
end

function sample_mode_roi(sampler::CurrentSampler, box::BoxROI, sampling_region::SVector{4, Integer})
    if sampler.valid_region == sampling_region
        sampler.sampling_region = sampling_region
    else
        sampler.sampling_region = SVector{4, Integer}(max(sampler.valid_region[1], sampling_region[1]), max(sampler.valid_region[2],sampling_region[2]),
        min(sampler.valid_region[3], sampling_region[3]), min(sampler.valid_region[4], sampling_region[4]))
    end

    if sampler.mode == :positive
        positive_sample = box.img[sampler.tracked_patch[1]:sampler.tracked_patch[3],
         sampler.tracked_patch[2]:sampler.tracked_patch[4]]
        positive_sample_pos = SVector{2, Integer}(sampler.tracked_patch[1], sampler.tracked_patch[2])
        samples = fill((positive_sample, positive_sample_pos), (1, 4))

        return samples
    end

    height = sampler.tracked_patch[3] - sampler.tracked_patch[1] + 1
    width = sampler.tracked_patch[4] - sampler.tracked_patch[2] + 1

    tl_block = SVector{4, Integer}(sampler.sampling_region[1], sampler.sampling_region[2], sampler.sampling_region[1] + height - 1,
    sampler.sampling_region[2] + width - 1)
    tr_block = SVector{4, Integer}(sampler.sampling_region[1], sampler.sampling_region[4] - width + 1, sampler.sampling_region[1] + height - 1,
    sampler.sampling_region[4])
    bl_block = SVector{4, Integer}(sampler.sampling_region[3] - height + 1, sampler.sampling_region[2], sampler.sampling_region[3],
    sampler.sampling_region[2] + width - 1)
    br_block = SVector{4, Integer}(sampler.sampling_region[3] - height + 1, sampler.sampling_region[4] - width + 1, sampler.sampling_region[3],
    sampler.sampling_region[4])

    if sampler.mode == :negative
        tl_sample = box.img[tl_block[1]:tl_block[3], tl_block[2]:tl_block[4]]
        tr_sample = box.img[tr_block[1]:tr_block[3], tr_block[2]:tr_block[4]]
        bl_sample = box.img[bl_block[1]:bl_block[3], bl_block[2]:bl_block[4]]
        br_sample = box.img[br_block[1]:br_block[3], br_block[2]:br_block[4]]
        samples = [(tl_sample, SVector{2, Integer}(tl_block[1:2])) (tr_sample, SVector{2, Integer}(tr_block[1:2])) (bl_sample, SVector{2, Integer}(bl_block[1:2])) (br_sample, SVector{2, Integer}(br_block[1:2]))]

        return samples
    end

    if sampler.mode == :classify
        step_row = max(1, floor(Int, (1 - sampler.overlap)*height + 0.5))
        step_col = max(1, floor(Int, (1 - sampler.overlap)*width + 0.5))
        max_row = sampler.sampling_region[3] - sampler.sampling_region[1] - height + 1
        max_col = sampler.sampling_region[4] - sampler.sampling_region[2] - width + 1

        samples = Array{Tuple{Array{T, 2}, SVector{2, Integer}}} where T <: Integer
        for i = 1:step_col:max_col
            for j = 1:step_row:max_row
                push!(samples, (box.img[j + sampler.sampling_region[1] - 1:j + sampler.sampling_region[1] + height - 2,
                i + sampler.sampling_region[2] - 1:i + sampler.sampling_region[2] + width - 2],
                SVector{2, Integer}(j + sampler.sampling_region[1]-1, i + sampler.sampling_region[2]-1)))
            end
        end

        return samples
    else
        error("Incorrect mode.")
    end

end
