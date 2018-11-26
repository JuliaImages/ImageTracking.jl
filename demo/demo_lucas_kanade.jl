using Images, TestImages, StaticArrays, ImageTracking, ImageView, LinearAlgebra, CoordinateTransformations, Gtk.ShortNames
#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#
img1 = load("demo/table1.jpg")
img2 = load("demo/table2.jpg")

corners = imcorner(img1, method=shi_tomasi)
I = findall(!iszero, corners)
r, c = (getindex.(I, 1), getindex.(I, 2))
points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)


algorithm = LucasKanade(20, window_size = 11,
                            pyramid_levels = 4,
                            eigenvalue_threshold = 0.000001)
flow, indicator = optical_flow(Gray{Float32}.(img1), Gray{Float32}.(img2),points, algorithm)

# Keep the subset of points that were successfully tracked and determine
# correspondences.
valid_points = points[indicator]
valid_flow = flow[indicator]
valid_correspondence = map((x,Δx)-> x+Δx, valid_points, valid_flow)

# Convert (row,columns) to (x,y) convention and round to nearest integer.
pts0 = map(x-> round.(Int,(last(x),first(x))), points)
pts1 = map(x-> round.(Int,(last(x),first(x))), valid_points)
pts2 = map(x-> round.(Int,(last(x),first(x))), valid_correspondence)
lines = map((p1, p2) -> (p1,p2), pts1, pts2)

# Visualise the optical flow. Red lines demarcate optical flow on the keypoints
# that were successfully tracked.
guidict = imshow(img1)
idx2 = annotate!(guidict, AnnotationLines(lines, linewidth=2.0, color=RGB(1,0,0), coord_order="xyxy"))
