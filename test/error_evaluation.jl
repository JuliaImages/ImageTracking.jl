# We output a message after loading each package to work around a
# ten-minute timeout limit on Travis. Travis assumes the tests have hung
# if the interval between printing something to stdio exceeds ten minutes.
using Images
@info "Finished loading Images package."
using StaticArrays
@info "Finished loading StaticArrays package."

@testset "Error Evaluation" begin
    @info "Running flow evaluation test."

    img1 = load("../demo/car2.jpg")
    img2 = load("../demo/car1.jpg") 

    algorithm = Farneback(50, estimation_window = 11,
                         σ_estimation_window = 9.0,
                         expansion_window = 6,
                         σ_expansion_window = 5.0)

    @time flow = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2), algorithm)

    # Endpoint error statistics
    @time endpoint_error = evaluate_flow_error(flow, flow, EndpointError())
    @time ep_mean, ep_sd, ep_RX, ep_AX = calculate_statistics(endpoint_error)

    # Angular error statistics
    @time angular_error = evaluate_flow_error(flow, flow, AngularError())
    @time ang_mean, ang_sd, ang_RX, ang_AX = calculate_statistics(angular_error)

    @test ep_mean == 0
    @test ep_sd == 0
    @test ang_mean == 0
    @test ang_sd == 0
    @test ep_RX[0.5] == 0
    @test ep_AX[0.5] < 1e-5
    @test ang_RX[0.5] == 0
    @test ang_AX[0.5] < 1e-5
end;
