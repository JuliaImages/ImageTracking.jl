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

hsv = visualize_flow(ColorBased(), flow, convention="row_col")

imshow(RGB.(hsv))
save("./optical_flow_farneback.jpg", hsv)
