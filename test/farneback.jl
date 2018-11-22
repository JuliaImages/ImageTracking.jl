using Images, TestImages, StaticArrays, OffsetArrays, Random, CoordinateTransformations

function evaluate_error(dims, flow::Array{SVector{2, Float64}, 2},  Δ,  tol)
    error_count = 0
    maximum_error = 0
    for r = 1:first(dims)
        for c = 1:last(dims)
            δ = flow[r,c]
            if abs(δ[1] - Δ[1]) > tol || abs(δ[2] - Δ[2]) > tol
                    error_count += 1
            end
            error = sqrt( (δ[1] - Δ[1])^2 + (δ[2] - Δ[2])^2 )
            if error > maximum_error
                maximum_error = error
            end
        end
    end
    return error_count, maximum_error
end

@testset "Farneback" begin
    @testset "Polynomial Expansion" begin
        img = Gray{Float64}.(testimage("mandrill"))
        A, B, C = polynomial_expansion(ConvolutionImplementation(), img, 9, 2)
        P, Q, R = polynomial_expansion(MatrixImplementation(), img, 9, 2)
        @test sum(abs.(vec(A)-vec(P))) ≈  0 atol=1e-9
        @test sum(abs.(vec(B)-vec(Q))) ≈  0 atol=1e-9
        @test sum(abs.(vec(C)-vec(R))) ≈  0 atol=1e-9

    end

    @testset "Horizontal Motion" begin
        maximum_percentage_error = 7.5
        img1 = Gray{Float64}.(testimage("mandrill"))

        tol = 0.3
        Δ = (0.0, 3.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        flow = optical_flow(img1, img2, Farneback(7, 39, 6.0, 11, 1.5))

        error_count, maximum_error = evaluate_error(size(img1), flow, Δ, tol)
        percentage_error = (error_count / prod(size(img1))) * 100

        @info "Case: Horizontal Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
        @test maximum_error < 4

        # Same as above, except that we pass in an initial displacement field.
        displacement = Array{SVector{2, Float64}, 2}(undef,size(img1))
        for i in eachindex(displacement)
                displacement[i] = SVector{2, Float64}(0.0, 0.0)
        end
        flow = optical_flow!(img1, img2, displacement, Farneback(7, 39, 6.0, 11, 1.5))

        error_count, maximum_error = evaluate_error(size(img1), flow, Δ, tol)
        percentage_error = (error_count / prod(size(img1))) * 100

        @info "Case: Horizontal Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
        @test maximum_error < 4
    end

    @testset "Vertical Motion" begin
        maximum_percentage_error = 7.5
        img1 = Gray{Float64}.(testimage("mandrill"))

        tol = 0.3
        Δ = (3.0, 0.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        flow = optical_flow(img1, img2, Farneback(7, 39, 6.0, 11, 1.5))

        error_count, maximum_error = evaluate_error(size(img1), flow, Δ, tol)
        percentage_error = (error_count / prod(size(img1))) * 100

        @info "Case: Vertical Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
        @test maximum_error < 4

        # Same as above, except that we pass in an initial displacement field.
        displacement = Array{SVector{2, Float64}, 2}(undef,size(img1))
        for i in eachindex(displacement)
                displacement[i] = SVector{2, Float64}(0.0, 0.0)
        end
        flow = optical_flow!(img1, img2, displacement, Farneback(7, 39, 6.0, 11, 1.5))

        error_count, maximum_error = evaluate_error(size(img1), flow, Δ, tol)
        percentage_error = (error_count / prod(size(img1))) * 100

        @info "Case: Vertical Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
        @test maximum_error < 4
    end

    @testset "Combined Motion" begin
        maximum_percentage_error = 13
        img1 = Gray{Float64}.(testimage("mandrill"))

        tol = 0.3
        Δ = (3.0, -1.0)
        trans = Translation(-Δ[1], -Δ[2])
        img2 = warp(img1, trans, axes(img1))

        flow = optical_flow(img1, img2, Farneback(7, 39, 6.0, 11, 1.5))

        error_count, maximum_error = evaluate_error(size(img1), flow, Δ, tol)
        percentage_error = (error_count / prod(size(img1))) * 100

        @info "Case: Combined Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
        @test maximum_error < 4

        # Same as above, except that we pass in an initial displacement field.
        displacement = Array{SVector{2, Float64}, 2}(undef,size(img1))
        for i in eachindex(displacement)
                displacement[i] = SVector{2, Float64}(0.0, 0.0)
        end
        flow = optical_flow!(img1, img2, displacement, Farneback(7, 39, 6.0, 11, 1.5))

        error_count, maximum_error = evaluate_error(size(img1), flow, Δ, tol)
        percentage_error = (error_count / prod(size(img1))) * 100

        @info "Case: Combined Motion", "Percentage Error: $(percentage_error)", "Maximum Error: $(maximum_error)"
        @test percentage_error  < maximum_percentage_error
        @test maximum_error < 4

    end

end;
