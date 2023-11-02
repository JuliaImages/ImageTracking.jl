struct LKPyramid
    layers::Vector{Matrix{Gray{Float64}}}

    Iy::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
    Ix::Union{Nothing, Vector{Matrix{Gray{Float64}}}}

    Iyy::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
    Ixx::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
    Iyx::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
end

function LKPyramid(image, levels; downsample = 2, σ = 1.0, compute_gradients::Bool = true)
    pyramid = gaussian_pyramid(image, levels, downsample, σ)
    !compute_gradients &&
        return LKPyramid(pyramid, nothing, nothing, nothing, nothing, nothing)

    total_levels = levels + 1
    Iy = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Ix = Vector{Matrix{Gray{Float64}}}(undef, total_levels)

    Iyy = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Ixx = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Iyx = Vector{Matrix{Gray{Float64}}}(undef, total_levels)

    filling = Fill(zero(eltype(pyramid[1])))
    for (i, layer) in enumerate(pyramid)
        Iy[i], Ix[i] = imgradients(layer, KernelFactors.scharr, filling)
        Iyy[i], Ixx[i], Iyx[i] = compute_partial_derivatives(Iy[i], Ix[i])
    end
    LKPyramid(pyramid, Iy, Ix, Iyy, Ixx, Iyx)
end

function optflow(
    first_img::AbstractMatrix{T}, second_img::AbstractMatrix{T},
    points::Vector{SVector{2, Float64}}, algorithm::LucasKanade,
) where T <: Gray
    displacement = fill(SVector{2, Float64}(0.0, 0.0), length(points))
    first_pyramid = LKPyramid(first_img, algorithm.pyramid_levels)
    second_pyramid = LKPyramid(
        second_img, algorithm.pyramid_levels; compute_gradients=false,
    )
    optflow!(first_pyramid, second_pyramid, points, displacement, algorithm)
end

function optflow!(
    first_img::AbstractMatrix{T}, second_img::AbstractMatrix{T},
    points::Vector{SVector{2, Float64}}, displacement::Vector{SVector{2, Float64}},
    algorithm::LucasKanade,
) where T <: Gray
    first_pyramid = LKPyramid(first_img, algorithm.pyramid_levels)
    second_pyramid = LKPyramid(
        second_img, algorithm.pyramid_levels; compute_gradients=false,
    )
    optflow!(first_pyramid, second_pyramid, points, displacement, algorithm)
end

function optflow!(
    first_pyramid::LKPyramid, second_pyramid::LKPyramid,
    points::Vector{SVector{2, Float64}}, displacement::Vector{SVector{2, Float64}},
    algorithm::LucasKanade,
)
    has_enough_layers =
        length(first_pyramid.layers) > algorithm.pyramid_levels &&
        length(second_pyramid.layers) > algorithm.pyramid_levels
    has_enough_layers || throw("Not enough layers in pyramids.")

    n_points = length(points)
    status = trues(n_points)

    window = algorithm.window_size
    window2x = 2 * window

    for level in (algorithm.pyramid_levels + 1):-1:1
        level_resolution = axes(first_pyramid.layers[level])
        # Interpolate layer to get sub-pixel precision.
        # We never go out-of-bound, so there is no need to extrapolate.
        interploated_layer = interpolate(second_pyramid.layers[level], BSpline(Linear()))

        for n in 1:n_points
            @inbounds !status[n] && continue

            point = get_pyramid_coordinate(@inbounds(points[n]), level)
            offsets = get_offsets(point, point, window, level_resolution)
            grid = get_grid(point, offsets)

            G_inv, min_eigenvalue = compute_spatial_gradient(first_pyramid, grid, level)
            if min_eigenvalue < algorithm.eigenvalue_threshold
                @inbounds status[n] = false
                continue
            end

            pyramid_contribution = SVector{2}(0.0, 0.0)
            for _ in 1:algorithm.iterations
                putative_flow = @inbounds displacement[n] + pyramid_contribution
                putative_correspondence = point + putative_flow
                if !lies_in(level_resolution, putative_correspondence)
                    @inbounds status[n] = false
                    break
                end

                new_offsets = get_offsets(
                    point, putative_correspondence, window, level_resolution,
                )
                # Recalculate gradient only if the offset changes.
                if new_offsets != offsets
                    offsets = new_offsets
                    grid = get_grid(point, offsets)
                    G_inv, min_eigenvalue = compute_spatial_gradient(
                        first_pyramid, grid, level,
                    )
                    if min_eigenvalue < algorithm.eigenvalue_threshold
                        @inbounds status[n] = false
                        break
                    end
                else
                    offsets = new_offsets
                    grid = get_grid(point, offsets)
                end

                estimated_flow = compute_flow_vector(
                    putative_correspondence,
                    first_pyramid, interploated_layer, level,
                    grid, offsets, G_inv,
                )
                # Epsilon termination criteria.
                abs(estimated_flow[1]) < algorithm.ϵ &&
                    abs(estimated_flow[2]) < algorithm.ϵ && break

                pyramid_contribution += estimated_flow
                # Check if tracked point is out of image bounds.
                if !lies_in(level_resolution, point + pyramid_contribution)
                    @inbounds status[n] = false
                    break
                end
            end
            @inbounds begin
            # Check if flow is too big.
            if status[n] && is_lost(displacement[n], window2x)
                status[n] = false
            end
            if status[n]
                displacement[n] += pyramid_contribution
                level != 1 && (displacement[n] *= 2.0)
            end
            end
        end
    end

    displacement, status
end

# TODO: Images.integral_image has been deprecated in favor of IntegralArrays.jl
# Included here to get this package working with newer versions of Images.jl
# Should this package use IntegralArrays.jl like the depreciation notes suggest?

accum(::Type{T}) where {T<:Integer} = Int
accum(::Type{Float32})    = Float32
accum(::Type{T}) where {T<:Real} = Float64
accum(::Type{Gray{T}}) where {T} = accum(T)

"""
```
integral_img = integral_image(img)
```

Returns the integral image of an image. The integral image is calculated by assigning
to each pixel the sum of all pixels above it and to its left, i.e. the rectangle from
(1, 1) to the pixel. An integral image is a data structure which helps in efficient
calculation of sum of pixels in a rectangular subset of an image. See `boxdiff` for more
information.
"""
function integral_image(img::AbstractArray)
    integral_img = Array{accum(eltype(img))}(undef, size(img))
    sd = coords_spatial(img)
    cumsum!(integral_img, img, dims=sd[1])
    for i = 2:length(sd)
        cumsum!(integral_img, integral_img, dims=sd[i])
    end
    integral_img
end

"""
```
sum = boxdiff(integral_image, ytop:ybot, xtop:xbot)
sum = boxdiff(integral_image, CartesianIndex(tl_y, tl_x), CartesianIndex(br_y, br_x))
sum = boxdiff(integral_image, tl_y, tl_x, br_y, br_x)
```

An integral image is a data structure which helps in efficient calculation of sum of pixels in
a rectangular subset of an image. It stores at each pixel the sum of all pixels above it and to
its left. The sum of a window in an image can be directly calculated using four array
references of the integral image, irrespective of the size of the window, given the `yrange` and
`xrange` of the window. Given an integral image -

        A - - - - - - B -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        C * * * * * * D -
        - - - - - - - - -

The sum of pixels in the area denoted by * is given by S = D + A - B - C.
"""
boxdiff(int_img::AbstractArray{T, 2}, y::UnitRange, x::UnitRange) where {T} =
    boxdiff(int_img, y.start, x.start, y.stop, x.stop)

boxdiff(int_img::AbstractArray{T, 2}, tl::CartesianIndex, br::CartesianIndex) where {T} =
    boxdiff(int_img, tl[1], tl[2], br[1], br[2])

function boxdiff(int_img::AbstractArray{T, 2}, tl_y::Integer, tl_x::Integer, br_y::Integer, br_x::Integer) where T
    sum = int_img[br_y, br_x]
    sum -= tl_x > 1 ? int_img[br_y, tl_x - 1] : zero(T)
    sum -= tl_y > 1 ? int_img[tl_y - 1, br_x] : zero(T)
    sum += tl_y > 1 && tl_x > 1 ? int_img[tl_y - 1, tl_x - 1] : zero(T)
    sum
end

function compute_partial_derivatives(Iy, Ix; σ = 4)
    kernel = KernelFactors.gaussian(σ)
    kernel_factors = kernelfactors((kernel, kernel))

    squared = typeof(Iy)(undef, size(Iy))
    filtered = typeof(Iy)(undef, size(Iy))

    squared .= Iy .* Iy
    imfilter!(filtered, squared, kernel_factors)
    Iyy_integral_table = integral_image(filtered)

    squared .= Ix .* Ix
    imfilter!(filtered, squared, kernel_factors)
    Ixx_integral_table = integral_image(filtered)

    squared .= Iy .* Ix
    imfilter!(filtered, squared, kernel_factors)
    Iyx_integral_table = integral_image(filtered)

    Iyy_integral_table, Ixx_integral_table, Iyx_integral_table
end

function _compute_spatial_gradient(
    grid, Iyy_integral, Iyx_integral, Ixx_integral,
)
    sum_Iyy = boxdiff(Iyy_integral, grid[1], grid[2])
    sum_Ixx = boxdiff(Ixx_integral, grid[1], grid[2])
    sum_Iyx = boxdiff(Iyx_integral, grid[1], grid[2])
    SMatrix{2, 2, Float64}(sum_Iyy, sum_Iyx, sum_Iyx, sum_Ixx)
end

function compute_spatial_gradient(pyramid::LKPyramid, grid, level)
    G = _compute_spatial_gradient(
        grid, pyramid.Iyy[level], pyramid.Iyx[level], pyramid.Ixx[level],
    )
    U, S, V = svd2x2(G)
    G_inv = pinv2x2(U, S, V)
    min_eigenvalue = min(S[1, 1], S[2, 2]) / prod(length.(grid))

    G_inv, min_eigenvalue
end

function prepare_linear_system(corresponding_point, A, Iy, Ix, offsets, B)
    P, Q = size(A)
    by, bx = 0.0, 0.0

    @inbounds for q in 1:Q, p in 1:P
        r = corresponding_point[1] + offsets[1][p]
        c = corresponding_point[2] + offsets[2][q]

        δI = A[p, q] - B(r, c)
        by += δI * Iy[p, q]
        bx += δI * Ix[p, q]
    end

    SVector{2, Float64}(by, bx)
end

function compute_flow_vector(
    corresponding_point,
    first_pyramid::LKPyramid, etp, level,
    grid, offsets, G_inv,
)
    b = prepare_linear_system(
        corresponding_point,
        view(first_pyramid.layers[level], grid[1], grid[2]),
        view(first_pyramid.Iy[level], grid[1], grid[2]),
        view(first_pyramid.Ix[level], grid[1], grid[2]),
        offsets, etp,
    )
    G_inv * b
end

@inline is_lost(displacement, window) =
    @inbounds(displacement[1] > window || displacement[2] > window)

@inline lies_in(area, point) = @inbounds(
    first(area[1]) ≤ point[1] ≤ last(area[1]) &&
    first(area[2]) ≤ point[2] ≤ last(area[2])
)

"""
# Arguments
- `level`: Level of the pyramid in `[1, levels]` range.
"""
@inline get_pyramid_coordinate(point, level) = floor.(Int, point ./ 2 ^ (level - 1))

function get_offsets(point, new_point, window, image_axes)
    rows, cols = image_axes

    @inbounds begin
    up = floor(Int, min(window, min(point[1], new_point[1]) - first(rows)))
    down = floor(Int, min(window, last(rows) - max(point[1], new_point[1])))
    left = floor(Int, min(window, min(point[2], new_point[2]) - first(cols)))
    right = floor(Int, min(window, last(cols) - max(point[2], new_point[2])))
    end

    (-up:down, -left:right)
end

function get_grid(point, offsets)
    map(point, offsets) do p, o
        p+first(o):p+last(o)
    end
end
