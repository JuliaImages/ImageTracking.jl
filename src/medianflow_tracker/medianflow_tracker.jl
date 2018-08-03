mutable struct TrackerMedianFlow{I <: Int, F <: Float64} <: Tracker
    #Initialized
    points_in_grid::I
    window_size::I
    max_level::I
    termination_criteria::I
    window_size_for_ncc::I
    max_median_of_displacement::F

    #Uninitialized
    model::TrackerModel

    TrackerMedianFlow(points_in_grid::I = 15, window_size::I = 11, max_level::I = 4, termination_criteria::I = 20, window_size_for_ncc::I = 30, max_median_of_displacement::F = 10.0) where {I <: Int, F <: Float64} = new{I, F}(
                      points_in_grid, window_size, max_level, termination_criteria, window_size_for_ncc, max_median_of_displacement)
end

function init_impl(tracker::TrackerMedianFlow{}, image::Array{T, 2}, bounding_box::MVector{4, Int}) where T
    tracker.model = TrackerMedianFlowModel(image, bounding_box)
end

function update_impl(tracker::TrackerMedianFlow{}, image::Array{T, 2}) where T
    prev_img = tracker.model.image
    bounding_box = tracker.model.bounding_box
    next_img = image

    gray_prev_img = Gray.(prev_img)
    gray_new_img = Gray.(next_img)

    points_to_track = Vector{SVector{2, Int}}()
    for i = 1:tracker.points_in_grid
        for j = 1:tracker.points_in_grid
            push!(points_to_track, SVector{2}(round(Int, bounding_box[1] + (i+0.5)*(bounding_box[3]/tracker.points_in_grid)), round(Int, bounding_box[2] + (j+0.5)*(bounding_box[4]/tracker.points_in_grid))))
        end
    end

    flow, status, err = optical_flow(gray_prev_img, gray_new_img, LK(points_to_track, [SVector{2}(0.0,0.0)], tracker.window_size, tracker.max_level, false, tracker.termination_criteria))
    new_points = points_to_track .+ flow
    good_points = findn(status)
    points_to_track = points_to_track[good_points]
    new_points = new_points[good_points]

    filter_status = MVector{length(good_points)}(trues(good_points))
    filter_status = calc_fb_error(tracker, prev_img, next_img, points_to_track, new_points, filter_status)
    filter_status = calc_ncc_error(tracker, prev_img, next_img, points_to_track, new_points, MVector{length(filter_status)}(filter_status))

    good_points = findn(filter_status)
    points_to_track = points_to_track[good_points]
    new_points = new_points[good_points]

    difference = new_points .- points_to_track
    displacement, bounding_box = get_displacement(tracker, points_to_track, new_points, bounding_box)
    final_disp = Array{SVector{2, Float64}, 1}()
    for i = 1:length(difference)
        push!(final_disp, difference[i] .- displacement)
    end
    displacements = sqrt.(dot.(final_disp,final_disp))
    median_displacements = median(displacements)
    assert(median_displacements < tracker.max_median_of_displacement)

    tracker.model.image = image
    tracker.model.bounding_box = bounding_box

    return bounding_box
end

@inline function get_y(point::SVector{2, T}) where T
    return point[1]
end

@inline function get_x(point::SVector{2, T}) where T
    return point[2]
end

function get_displacement(tracker::TrackerMedianFlow{}, old_points::Array{SVector{2, Int}, 1}, new_points::Array{SVector{2, Float64}, 1}, bounding_box::MVector{4, Int})
    new_center = MVector{2}(round(Int, (bounding_box[1] + bounding_box[3] + 1)/2), round(Int, (bounding_box[2] + bounding_box[4] + 1)/2))
    new_bounding_box = MVector{4}(zeros(Int, 4))

    if length(old_points) == 1
        new_bounding_box[1] = bounding_box[1] + new_points[1][1] - old_points[1][1]
        new_bounding_box[2] = bounding_box[2] + new_points[1][2] - old_points[1][2]
        new_bounding_box[3] = new_bounding_box[1] + bounding_box[3] - bounding_box[1] + 1
        new_bounding_box[4] = new_bounding_box[2] + bounding_box[4] - bounding_box[2] + 1
        displacement = MVector{2}(new_points[1][1] - old_points[1][1], new_points[1][2] - old_points[1][2])
        return displacement, new_bounding_box
    end

    location_shift = new_points .- old_points
    y_shift = median(get_y.(location_shift))
    x_shift = median(get_x.(location_shift))
    new_center .+= MVector{2}(round(Int, y_shift), round(Int, x_shift))
    displacement = MVector{2}(round(Int, y_shift), round(Int, x_shift))

    n = length(old_points)
    scale_buffer = MVector{round(Int, n*(n-1)/2), Float64}()
    counter = 1
    for i = 1:length(old_points)
        for j = 1:i-1
            new_change = norm(new_points[i] .- new_points[j])
            old_change = norm(old_points[i] .- old_points[j])
            scale_buffer[counter] = (old_change == 0.0)? 0.0:(new_change/old_change)
            counter += 1
        end
    end
    scale = median(scale_buffer)
    new_bounding_box[1] = round(Int, new_center[1] - scale*(bounding_box[3] - bounding_box[1] + 1)/2)
    new_bounding_box[2] = round(Int, new_center[2] - scale*(bounding_box[4] - bounding_box[2] + 1)/2)
    new_bounding_box[3] = round(Int, new_bounding_box[1] + scale*(bounding_box[3] - bounding_box[1] + 1))
    new_bounding_box[4] = round(Int, new_bounding_box[2] + scale*(bounding_box[4] - bounding_box[2] + 1))

    return displacement, new_bounding_box
end

function calc_fb_error(tracker::TrackerMedianFlow, prev_img::Array{T, 2}, next_img::Array{T, 2}, old_points::Array{SVector{2, Int}, 1}, new_points::Array{SVector{2, Float64}, 1}, status::MVector{N, Bool}) where N where T
    #TODO: Add Float64 support for input array in LK optical flow
    flow, status, err = optical_flow(prev_img, next_img, LK(round(Int, new_points), [SVector{2}(0.0,0.0)], tracker.window_size, tracker.max_level, false, tracker.termination_criteria))

    fb_error = norm.(old_points .- flow)
    median_fb_error = median(fb_error)
    status .= status .& (fb_error .<= median_fb_error)
    return status
end

function get_patch(image::Array{T, 2}, patch_size::MVector{2, Int}, patch_center::SVector{2, I}) where {T, I <: Real}
    roi_strat_corner = MVector{2, Int}(round(Int, patch_center[1] - patch_size[1]/2), round(Int, patch_center[2] - patch_size[2]/2))
    patch_rect = MVector{4, Int}(roi_strat_corner[1], roi_strat_corner[2], roi_strat_corner[1] + patch_size[1] - 1, roi_strat_corner[2] + patch_size[2] - 1)

    if patch_rect[1] >= 1 && patch_rect[2] >= 1 && patch_rect[3] <= size(image)[1] && patch_rect[4] <= size(image)[2]
        patch = image[patch_rect[1]:patch_rect[3], patch_rect[2]:patch_rect[4]]
    else
        itp = interpolate(image, BSpline(Quadratic(Flat())), OnGrid())
        etp = extrapolate(itp, zero(eltype(image)))
        patch = etp[patch_rect[1]:patch_rect[3], patch_rect[2]:patch_rect[4]]
    end

    return patch
end

function calc_ncc_error(tracker::TrackerMedianFlow, prev_img::Array{T, 2}, next_img::Array{T, 2}, old_points::Array{SVector{2, Int}, 1}, new_points::Array{SVector{2, Float64}, 1}, status::MVector{N, Bool}) where N where T
    ncc_error = MVector{length(old_points), Float64}()
    for i = 1:length(old_points)
        patch_1 = get_patch(prev_img, MVector{2}(tracker.window_size_for_ncc, tracker.window_size_for_ncc), old_points[i])
        patch_2 = get_patch(next_img, MVector{2}(tracker.window_size_for_ncc, tracker.window_size_for_ncc), new_points[i])

        temp = ncc(Float64.(patch_1), Float64.(patch_2))
        #TODO: Use DataArrays.jl to handle missing data
        ncc_error[i] = isnan(temp) ? 0 : temp
    end

    median_ncc_error = median(ncc_error)
    status .= status .& (ncc_error .<= median_ncc_error)
    return status
end
