using Images
using TestImages
using StaticArrays
using OffsetArrays
using Random
using CoordinateTransformations

function evaluate_error(dims, flow::Array{SVector{2, Float64}, 1}, Δ, tol)
    error_count = 0
    maximum_error = 0.0
    for i in eachindex(flow)
        δ = flow[i]
        if abs(δ[1] - Δ[1]) > tol || abs(δ[2] - Δ[2]) > tol
            error_count += 1
        end
        error = sqrt( (δ[1] - Δ[1])^2 + (δ[2] - Δ[2])^2 )
        if error > maximum_error
            maximum_error = error
        end
    end
    return error_count, maximum_error
end

function get_keypoints(image, n_points)
    corners = imcorner(image, method=shi_tomasi)
    I = CartesianIndices(corners)[corners .!= 0]
    r, c = (getindex.(I, 1), getindex.(I, 2))
    points = map((ri, ci) -> SVector{2, Float64}(ri, ci), r, c)
    rand(points, n_points)
end

@testset "Lucas-Kanade" begin
    @testset "Horizontal Motion" begin
        @info "Running Horizontal Motion test."
        Random.seed!(9876)

        tol = 1e-1
        maximum_percentage_error = 7
        number_test_pts = 500
        img1 = Gray{Float64}.(testimage("mandrill"))
        algorithm = LucasKanade()

        Δ = (0.0, 3.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        points = get_keypoints(img1, number_test_pts)

        flow, status = optical_flow(img1, img2, points, algorithm)
        error_count, maximum_error = evaluate_error(size(img1), flow[status], Δ, tol)
        percentage_error = (error_count / sum(status)) * 100
        @info "Case: Horizontal Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error < maximum_percentage_error

        # Same as above, except that we pass in an initial displacement field.
        displacement = fill(SVector{2, Float64}(0.0, 0.0), length(points))
        flow, status = optical_flow(img1, img2, points, displacement, algorithm)
        error_count, maximum_error = evaluate_error(size(img1), flow[status], Δ, tol)
        percentage_error = (error_count / sum(status)) * 100
        @info "Case: Horizontal Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error < maximum_percentage_error
    end

    @testset "Vertical Motion" begin
        @info "Running Vertical Motion test."
        Random.seed!(9876)

        tol = 1e-1
        maximum_percentage_error = 7
        number_test_pts = 500
        img1 = Gray{Float64}.(testimage("mandrill"))
        algorithm = LucasKanade()

        Δ = (3.0, 0.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        points = get_keypoints(img1, number_test_pts)

        flow, status_array = optical_flow(img1, img2, points, algorithm)
        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100
        @info "Case: Vertical Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error < maximum_percentage_error

        # Same as above, except that we pass in an initial displacement field.
        displacement = fill(SVector{2, Float64}(0.0, 0.0), length(points))
        flow, status_array = optical_flow(img1, img2, points, displacement, algorithm)
        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100
        @info "Case: Vertical Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error < maximum_percentage_error
    end

    @testset "Combined Motion" begin
        @info "Running Combined Motion test."
        Random.seed!(9876)

        tol = 1e-1
        maximum_percentage_error = 7
        number_test_pts = 500
        img1 = Gray{Float64}.(testimage("mandrill"))
        algorithm = LucasKanade()

        Δ = (3.0, -1.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        points = get_keypoints(img1, number_test_pts)

        flow, status_array = optical_flow(img1, img2, points, algorithm)
        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100
        @info "Case: Combined Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error < maximum_percentage_error

        # Same as above, except that we pass in an initial displacement field.
        displacement = fill(SVector{2, Float64}(0.0, 0.0), length(points))
        flow, status_array = optical_flow(img1, img2, points, displacement, algorithm)
        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100
        @info "Case: Combined Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error < maximum_percentage_error
    end
end
