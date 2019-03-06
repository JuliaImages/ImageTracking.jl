# We output a message after loading each package to work around a
# ten-minute timeout limit on Travis. Travis assumes the tests have hung
# if the interval between printing something to stdio exceeds ten minutes.
using Images
@info "Finished loading Images package."
using StaticArrays
@info "Finished loading StaticArrays package."



@testset "Visualization" begin
    @info "Running flow visualization test."
    flow = Array{SVector{2, Float64}, 2}(undef,(3, 2))

    flow[1] = SVector{2, Float64}(0.5, 0.5)
    flow[2] = SVector{2, Float64}(-0.4, -0.8)
    flow[3] = SVector{2, Float64}(-0.5, 0.3)
    flow[4] = SVector{2, Float64}(-0.4, -0.8) 
    flow[5] = SVector{2, Float64}(-0.5, 0.3)
    flow[6] = SVector{2, Float64}(0.5, 0.5)

    @time hsv = visualize_flow(ColorBased(), flow) 

    @test hsv[1] == hsv[6]
    @test hsv[2] == hsv[4]
    @test hsv[3] == hsv[5]

end;
