using ImageMagick
using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames
#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("car2.jpg")
img2 = load("car1.jpg")


algorithm = Farneback(50, estimation_window = 11,
                         σ_estimation_window = 9.0,
                         expansion_window = 6,
                         σ_expansion_window = 5.0)

flow = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2), algorithm)

# flow visualization
hsv = visualize_flow(flow, ColorBased(), convention="row_col")

# Endpoint error statistics
endpoint_error = evaluate_error(flow, flow, EndpointError())
ep_mean, ep_sd, ep_RX, ep_AX = calculate_statistics(endpoint_error)

# Angular error statistics
angular_error = evaluate_error(flow, flow, AngularError())
ang_mean, ang_sd, ang_RX, ang_AX = calculate_statistics(angular_error)

imshow(RGB.(hsv))
save("./optical_flow_farneback.jpg", hsv)

