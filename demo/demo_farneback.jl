using ImageMagick
using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames
#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("demo/car2.jpg")
img2 = load("demo/car1.jpg")

hsv = visualizeDenseFLowHSV(img1, img2)

imshow(RGB.(hsv))
save("./demo/optical_flow_farneback.jpg", hsv)