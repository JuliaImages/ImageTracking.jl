abstract type TrackerSampler end

function sample(sampler::TrackerSampler, image::Array{T, 2}, ROI::MVector{4, Int}) where T
    sample_impl(sampler, image, ROI)
end

#-------------------------------------------
# CURRENT STATE CENTERED SAMPLER ALGORITHM
# Used by MIL Tracker
#-------------------------------------------

mutable struct TS_Current_Sample_Centered{F <: Float64, I <: Int, S <: Symbol} <: TrackerSampler
    initial_inner_radius::F
    window_size::F
    initial_max_negatives::I
    inner_positive_radius::F
    tracking_max_positives::I
    tracking_max_negatives::I
    mode::S
    rng::MersenneTwister
end

TS_Current_Sample_Centered(initial_inner_radius::F = 3.0, window_size::F = 25.0, initial_max_negatives::I = 65, inner_positive_radius::F = 4.0, tracking_max_positives::I = 100000, tracking_max_negatives::I = 65, mode::S = :init_positive) where {F <: Float64, I <: Int, S <: Symbol} = TS_Current_Sample_Centered{F, I, S}(
                           initial_inner_radius, window_size, initial_max_negatives, inner_positive_radius, tracking_max_positives, tracking_max_negatives, mode, MersenneTwister(1234))

function sample_impl(sampler::TS_Current_Sample_Centered{}, image::Array{T, 2}, ROI::MVector{4, Int}) where T
    if sampler.mode == :init_positive
        in_rad = sampler.initial_inner_radius
        sample = sampling(sampler, image, ROI, in_rad)
    elseif sampler.mode == :init_negative
        in_rad = 2*sampler.window_size
        out_rad = 1.5*sampler.initial_inner_radius
        max_num = sampler.initial_max_negatives
        sample = sampling(sampler, image, ROI, in_rad, out_rad, max_num)
    elseif sampler.mode == :track_positive
        in_rad = sampler.inner_positive_radius
        out_rad = 0
        max_num = sampler.tracking_max_positives
        sample = sampling(sampler, image, ROI, in_rad, out_rad, max_num)
    elseif sampler.mode == :track_negative
        in_rad = 1.5*sampler.window_size
        out_rad = sampler.inner_positive_radius + 5
        max_num = sampler.tracking_max_negatives
        sample = sampling(sampler, image, ROI, in_rad, out_rad, max_num)
    else
        in_rad = sampler.window_size
        sample = sampling(sampler, image, ROI, in_rad)
    end

    return sample
end

function sampling(sampler::TS_Current_Sample_Centered{}, image::AbstractArray{T, 2}, ROI::MVector{4, Int}, in_rad::Float64, out_rad::Float64 = 0.0, max_num::Int = 1000000) where T
    height = ROI[3] - ROI[1] + 1
    width = ROI[4] - ROI[2] + 1

    rows = max(1, ROI[1] - floor(Int, in_rad) + 1):min(size(image)[1] - height - 1, ROI[1] + floor(Int, in_rad) + 1)
    cols = max(1, ROI[2] - floor(Int, in_rad) + 1):min(size(image)[2] - width - 1, ROI[2] + floor(Int, in_rad) + 1)

    sample = Array{Tuple{Array{T, 2}, MVector{2, Int}}, 1}()
    probability = max_num/((rows.stop-rows.start+1)*(cols.stop-cols.start+1))

    for i = cols
        for j = rows
            dist = ((ROI[1] - j)^2) + ((ROI[2] - i)^2)
            if rand(sampler.rng, Float64) < probability && dist < in_rad^2 && dist >= out_rad^2
                push!(sample, (image[j:j+height-1, i:i+width-1], MVector{2}(j, i)))
            end
        end
    end

    return sample
end
