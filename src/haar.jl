"""
```
features = haar_features(img, top_left, bottom_right, feat)
features = haar_features(img, top_left, bottom_right, feat, coordinates)
```

Returns an array containing the Haar-like features for the given Integral Image in the region specified by the points top_left and bottom_right.

Parameters:

 -  img               = The Integral Image for which the Haar-like features are to be found
 -  top_left          = The top and left most point of the region where the features are to be found
 -  bottom_right      = The bottom and right most point of the region where the features are to be found
 -  feat              = A symbol specifying the type of the Haar-like feature to be found

```
    Currently, feat can take 5 values:

    :x2 = Two rectangles along horizontal axis
    :y2 = Two rectangles along vertical axis
    :x3 = Three rectangles along horizontal axis
    :y3 = Three rectangles along vertical axis
    :xy4 = Four rectangles along horizontal and vertical axes
```

 -  coordinates       = The user can provide the coordinates of the rectangles if only certain Haar-like features are to found in the given region. The required format is a 3 Dimensional array as (f,r,4) where f = number_of_features, r = number_of_rectangles and 4 is for the coordinates of the top_left and bottom_right point of the rectangle (top_left_y, top_left_x, bottom_right_y, bottom_right_x). The default value is nothing, where all the features are found

## References

M. Oren, C. Papageorgiou, P. Sinha, E. Osuna and T. Poggio, "Pedestrian detection using wavelet templates," Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, San Juan, 1997, pp. 193-199.

P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 2001, pp. I-511-I-518 vol.1.

"""
function haar_features(img::AbstractArray{T, 2}, top_left::Array{I, 1}, bottom_right::Array{I, 1}, feat::Symbol, coordinates = nothing) where {T <: Union{Real, Color}, I <: Int}
    check_coordinates(top_left, bottom_right)
    nrect = check_feature_type(feat)

    if coordinates == nothing
        rectangle_features = haar_coordinates(bottom_right[1] - top_left[1] + 1, bottom_right[2] - top_left[2] + 1, feat)
    else
        a = Array{SVector{4,Int}}(nrect)
        for i = 1:nrect
            a[i] = SVector{4}(0,0,0,0)
        end
        rectangle_features = Array([a])
        for i = 1:size(coordinates)[1]
            temp = []
            for j = 1:size(coordinates)[2]
                push!(temp, SVector{4}(coordinates[i,j,1], coordinates[i,j,2], coordinates[i,j,3], coordinates[i,j,4]))
            end
            push!(rectangle_features, temp)
        end
        deleteat!(rectangle_features, 1)
    end
    if length(rectangle_features) > 0
        feats = length(rectangle_features)
        rects = length(rectangle_features[1])
    else
        println("No features of given type found in region!")
        feats = 0
        rects = 0
    end

    features = zeros(T, feats, rects)
    for i = 1:feats
        for j = 1:rects
            features[i,j] = boxdiff(img, rectangle_features[i][j][1]:rectangle_features[i][j][3], rectangle_features[i][j][2]:rectangle_features[i][j][4])
        end
    end

    output = (sum(features[:, 2:2:end], 2) - sum(features[:, 1:2:end], 2))[:]
end

"""
```
coordinates = haar_coordinates(height, width, feat)
```

Returns an array containing the coordinates of the Haar-like features of the specified type.

Parameters:

 -  height        = Height of the neighbourhood/window where the coordinates are computed
 -  width         = Width of the neighbourhood/window where the coordinates are computed
 -  feat          = A symbol specifying the type of the Haar-like feature to be found

```
    Currently, feat can take 5 values:

    :x2 = Two rectangles along horizontal axis
    :y2 = Two rectangles along vertical axis
    :x3 = Three rectangles along horizontal axis
    :y3 = Three rectangles along vertical axis
    :xy4 = Four rectangles along horizontal and vertical axes
```

"""
function haar_coordinates(h::Int, w::Int, feat::Symbol)
    nrect = check_feature_type(feat)

    a = Array{SVector{4,Int}}(nrect)
    for i = 1:nrect
        a[i] = SVector{4}(0,0,0,0)
    end
    rectangle_features = Array([a])

    for I = 1:h
        for J = 1:w
            for i = 2:h
                for j = 2:w
                    if feat == :x2 && I + i - 2 <= h && J + 2*j - 3 <= w
                        push!(rectangle_features, [SVector{4}(I, J, I + i - 2, J + j - 2), SVector{4}(I, J + j - 1, I + i - 2, J + 2*j - 3)])
                    elseif feat == :y2 && I + 2*i - 3 <= h && J + j - 2 <= w
                        push!(rectangle_features, [SVector{4}(I, J, I + i - 2, J + j - 2), SVector{4}(I + i - 1, J, I + 2*i - 3, J + j - 2)])
                    elseif feat == :x3 && I + i - 2 <= h && J + 3*j - 4 <= w
                        push!(rectangle_features, [SVector{4}(I, J, I + i - 2, J + j - 2), SVector{4}(I, J + j - 1, I + i - 2, J + 2*j - 3), SVector{4}(I, J + 2*j - 2, I + i - 2, J + 3*j - 4)])
                    elseif feat == :y3 && I + 3*i - 4 <= h && J + j  - 2 <= w
                        push!(rectangle_features, [SVector{4}(I, J, I + i - 2, J + j - 2), SVector{4}(I + i - 1, J, I + 2*i - 3, J + j - 2), SVector{4}(I + 2*i - 2, J, I + 3*i - 4, J + j - 2)])
                    elseif feat == :xy4 && I + 2*i - 3 <= h && J + 2*j - 3 <= w
                        push!(rectangle_features, [SVector{4}(I, J, I + i - 2, J + j - 2), SVector{4}(I, J + j - 1, I + i - 2, J + 2*j - 3), SVector{4}(I + i - 1, J + j - 1, I + 2*i - 3, J + 2*j - 3), SVector{4}(I + i - 1, J, I + 2*i - 3, J + j - 2)])
                    end
                end
            end
        end
    end
    deleteat!(rectangle_features, 1)

    return rectangle_features
end

function check_coordinates(top_left::Array{I, 1}, bottom_right::Array{I, 1}) where I <: Int
    if length(top_left) != 2 || length(bottom_right) != 2
        throw(ArgumentError("The top_left and bottom_right point must have only 2 coordinates."))
    end

    if bottom_right[1] < top_left[1]
        throw(ArgumentError("The bottom_right point must be lower than the top_left point."))
    end

    if bottom_right[2] < top_left[2]
        throw(ArgumentError("The bottom_right point must be towards the right of the top_left point."))
    end
end

function check_feature_type(feat::Symbol)
    if feat == :x2 || feat == :y2
        nrect = 2
    elseif feat == :x3 || feat == :y3
        nrect = 3
    elseif feat == :xy4
        nrect = 4
    else
        throw(ArgumentError("The type of the feature must be either :x2, :y2, :x3, :y3 or :xy4."))
        nrect = 0
    end

    return nrect
end
