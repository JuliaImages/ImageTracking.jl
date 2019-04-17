using ImageMagick
using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames
#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("./error_evaluation_demo_data/frame1.png")
img2 = load("./error_evaluation_demo_data/frame2.png")


algorithm = Farneback(50, estimation_window = 11,
                         σ_estimation_window = 9.0,
                         expansion_window = 6,
                         σ_expansion_window = 5.0)

estimated_flow = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2), algorithm)

# read the ground truth flow
file_path =  "./error_evaluation_demo_data/example.flo"
ground_truth_flow = read_flow_file(file_path)

# Endpoint error statistics
endpoint_error = evaluate_flow_error(ground_truth_flow, estimated_flow, EndpointError())
ep_mean, ep_sd, ep_RX, ep_AX = calculate_statistics(endpoint_error)
println("Endpoint error statistics :")
println("Mean = ", ep_mean, ", standard deviation = ", ep_sd, ", RX Dict = ", ep_RX, ", AX Dict = ", ep_AX)

# Angular error statistics
angular_error = evaluate_flow_error(ground_truth_flow, estimated_flow, AngularError())
ang_mean, ang_sd, ang_RX, ang_AX = calculate_statistics(angular_error)
println("Angular error statistics :")
println("Mean = ", ang_mean, ", standard deviation = ", ang_sd, ", RX Dict = ", ang_RX, ", AX Dict = ", ang_AX)

