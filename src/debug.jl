using Images, TestImages, StaticArrays, ImageTracking, Random, OffsetArrays, CoordinateTransformations, BenchmarkTools


img1 = Gray{Float64}.(testimage("mandrill"))
trans = Translation(0, -3.0)
img2 = warp(img1, trans, axes(img1))

corners = imcorner(img1, method=shi_tomasi)
I = findall(!iszero, corners)
r, c = (getindex.(I, 1), getindex.(I, 2))
points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)

flow, indicator  = optical_flow(img1, img2, points, LucasKanade( 11, 4, 20, 0.000001))

@btime flow, indicator  =  optical_flow($img1, $img2, $points, LucasKanade( 11, 4, 20, 0.000001))

(sum(indicator)/ length(indicator)) * 100

@code_warntype  optical_flow(img1, img2, points, LucasKanade( 11, 4, 20, 0.000001))
