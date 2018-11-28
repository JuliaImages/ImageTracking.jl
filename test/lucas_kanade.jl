# We output a message after loading each package to work around a
# ten-minute timeout limit on Travis. Travis assumes the tests have hung
# if the interval between printing something to stdio exceeds ten minutes.
using Images
@info "Finished loading Images package."
using TestImages
@info "Finished loading TestImages package."
using StaticArrays
@info "Finished loading StaticArrays package."
using OffsetArrays
@info "Finished loading OffsetArrays package."
using Random
@info "Finished loading Random package."
using CoordinateTransformations
@info "Finished loading CoordinateTransformations package."

function evaluate_error(dims, flow::Array{SVector{2, Float64}, 1},  Δ,  tol)
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

@testset "Lucas-Kanade" begin
    @testset "Horizontal Motion" begin
        @info "Running Horizontal Motion test."
        maximum_percentage_error = 7.5
        number_test_pts = 500
        img1 = Gray{Float64}.(testimage("mandrill"))

        tol = 0.3
        Δ = (0.0, 3.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        corners = imcorner(img1, method=shi_tomasi)
        I = findall(!iszero, corners)
        r, c = (getindex.(I, 1), getindex.(I, 2))
        points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)
        Random.seed!(9876)
        points = rand(points, (number_test_pts,))
        flow, status_array = optical_flow(img1, img2, points, LucasKanade(20, 11, 4, 0.000001))

        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100

        @info "Case: Horizontal Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error

        # Same as above, except that we pass in an initial displacement field.
        displacement = Array{SVector{2, Float64}, 1}(undef,length(points))
        for i in eachindex(displacement)
                displacement[i] = SVector{2, Float64}(0.0, 0.0)
        end
        flow, status_array = optical_flow(img1, img2, points, displacement, LucasKanade(20, 11, 4, 0.000001))

        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100

        @info "Case: Horizontal Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
    end

    @testset "Vertical Motion" begin
        @info "Running Vertical Motion test."
        maximum_percentage_error = 7.5
        number_test_pts = 500
        img1 = Gray{Float64}.(testimage("mandrill"))

        tol = 0.3
        Δ = (3.0, 0.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        corners = imcorner(img1, method=shi_tomasi)
        I = findall(!iszero, corners)
        r, c = (getindex.(I, 1), getindex.(I, 2))
        points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)
        Random.seed!(9876)
        points = rand(points, (number_test_pts,))
        flow, status_array = optical_flow(img1, img2, points, LucasKanade(20, 11, 4, 0.000001))

        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100

        @info "Case: Vertical Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error

        # Same as above, except that we pass in an initial displacement field.
        displacement = Array{SVector{2, Float64}, 1}(undef,length(points))
        for i in eachindex(displacement)
                displacement[i] = SVector{2, Float64}(0.0, 0.0)
        end
        flow, status_array = optical_flow(img1, img2, points, displacement, LucasKanade(20, 11, 4, 0.000001))

        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100

        @info "Case: Vertical Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
    end

    @testset "Combined Motion" begin
        @info "Running Combined Motion test."
        maximum_percentage_error = 13
        number_test_pts = 500
        img1 = Gray{Float64}.(testimage("mandrill"))

        tol = 0.3
        Δ = (3.0, -1.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        corners = imcorner(img1, method=shi_tomasi)
        I = findall(!iszero, corners)
        r, c = (getindex.(I, 1), getindex.(I, 2))
        points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)
        Random.seed!(9876)
        points = rand(points, (number_test_pts,))
        flow, status_array = optical_flow(img1, img2, points, LucasKanade(20, 11, 4, 0.000001))

        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100

        @info "Case: Combined Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error

        # Same as above, except that we pass in an initial displacement field.
        displacement = Array{SVector{2, Float64}, 1}(undef,length(points))
        for i in eachindex(displacement)
                displacement[i] = SVector{2, Float64}(0.0, 0.0)
        end
        flow, status_array = optical_flow(img1, img2, points, displacement, LucasKanade(20, 11, 4, 0.000001))

        error_count, maximum_error = evaluate_error(size(img1), flow[status_array], Δ, tol)
        percentage_error = (error_count / sum(status_array)) * 100

        @info "Case: Combined Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
    end

end;
