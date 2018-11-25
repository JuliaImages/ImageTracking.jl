using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames
#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("demo/car2.jpg")
img2 = load("demo/car1.jpg")

algorithm = Farneback(50, estimation_window = 11,
                         σ_estimation_window = 9.0,
                         expansion_window = 6,
                         σ_expansion_window = 5.0)
flow = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2), algorithm)

# Convert from (row,column) to (x,y) convention.
map!(x-> SVector(last(x),first(x)), flow, flow)

# Display optical flow as an image, with hue encoding the orientation and
# saturation encoding the relative magnitude.
max_norm = maximum(map(norm,flow))
normalised_flow = map(PolarFromCartesian(),flow / max_norm)
hsv = zeros(HSV{Float32},size(img1))
for i in eachindex(flow)
    hsv[i] = HSV((normalised_flow[i].θ + pi) * 180 / pi, normalised_flow[i].r, 1)
end
# Visualize the optical flow and save it to disk.
imshow(RGB.(hsv))
save("./demo/optical_flow_farneback.jpg", hsv)
