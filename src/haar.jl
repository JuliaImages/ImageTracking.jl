"""
```
features = haar_features(img, top_left, bottom_right, feat)
features = haar_features(img, top_left, bottom_right, feat, coordinates)
```

Returns an array containing the Haar-like features for the given Integral Image in the region specified by the points top_left and bottom_right.

The caller of the haar_features function specifies a rectangular region in the image by its top_left and bottom_right points and a particular
feature type, e.g. two rectangles side by side (:x2). This function then computes the values of all haar rectangular features for the particular
rectangle configuration for all possible locations, and all possible rectangle widths & heights if coordinates of certain specific features are not
provided. If they are provided then it only computes the values of those features whose coordinates are given. Calculating value of haar rectangular
feature corresponds to finding the difference of sums of all points in the different rectangles comprising the feature.

Parameters:

 -  img               = The Integral Image for which the Haar-like features are to be found
 -  top_left          = The top and left most point of the region where the features are to be found
 -  bottom_right      = The bottom and right most point of the region where the features are to be found
 -  feat              = A symbol specifying the type of the Haar-like feature to be found

```
    Currently, feat can take 5 values:

    :x2 = Two rectangles along horizontal axis
        +---------------------------------------+
        |                                       |
        | +-----+-----+                         |
        | |     |     |                         |
        | | -1  | +1  |  +---------->           |
        | |     |     |                         |
        | +-----+-----+                         |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :y2 = Two rectangles along vertical axis
        +---------------------------------------+
        |                                       |
        | +------------+                        |
        | |     -1     |                        |
        | +------------+  +---------->          |
        | |     +1     |                        |
        | +------------+                        |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :x3 = Three rectangles along horizontal axis
        +---------------------------------------+
        |                                       |
        | +-----+-----+-----+                   |
        | |     |     |     |                   |
        | | -1  | +1  | -1  |  +---------->     |
        | |     |     |     |                   |
        | +-----+-----+-----+                   |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :y3 = Three rectangles along vertical axis
        +---------------------------------------+
        |                                       |
        | +------------+                        |
        | |     -1     |                        |
        | +------------+                        |
        | |     +1     |  +---------->          |
        | +------------+                        |
        | |     -1     |                        |
        | +------------+                        |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :xy4 = Four rectangles along horizontal and vertical axes
        +---------------------------------------+
        |                                       |
        | +-----+-----+                         |
        | |     |     |                         |
        | | -1  | +1  |                         |
        | |     |     |                         |
        | +-----+-----+  +---------->           |
        | |     |     |                         |
        | | +1  | -1  |                         |
        | |     |     |                         |
        | +-----+-----+                         |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        +---------------------------------------+

    The +1 and -1 signs show which rectangles are subtracted and which are added to evaluate the final haar feature.

```

 -  coordinates       = The user can provide the coordinates of the rectangles if only certain Haar-like features are to found in the given region. The required format is a 3 Dimensional array as (f,r,4) where f = number_of_features, r = number_of_rectangles and 4 is for the coordinates of the top_left and bottom_right point of the rectangle (top_left_y, top_left_x, bottom_right_y, bottom_right_x). The default value is nothing, where all the features are found

## References

M. Oren, C. Papageorgiou, P. Sinha, E. Osuna and T. Poggio, "Pedestrian detection using wavelet templates," Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, San Juan, 1997, pp. 193-199.

P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 2001, pp. I-511-I-518 vol.1.

"""
function haar_features(img::AbstractArray{T, 2}, top_left::Array{I, 1}, bottom_right::Array{I, 1}, feat::Symbol, coordinates = nothing) where {T <: Union{Real, Color}, I <: Int}
    check_coordinates(top_left, bottom_right)
    rectangular_feature_coordinates = check_feature_type(feat)

    if coordinates == nothing
        rectangular_feature_coordinates = haar_coordinates(bottom_right[1] - top_left[1] + 1, bottom_right[2] - top_left[2] + 1, feat)
    else
        for i = 1:size(coordinates)[1]
            push!(rectangular_feature_coordinates, SMatrix{size(coordinates)[2], 4}(coordinates[i, :, :]))
        end
    end
    if length(rectangular_feature_coordinates) > 0
        num_feats = length(rectangular_feature_coordinates)
        num_rects = size(rectangular_feature_coordinates[1])[1]
    else
        println("No features of given type found in region!")
        num_feats = 0
        num_rects = 0
    end

    rectangular_features = zeros(T, num_feats, num_rects)
    for i = 1:num_feats
        a = rectangular_feature_coordinates[i]
        for j = 1:num_rects
            ytop = top_left[1] + a[j, 1] - 1
            ybot = top_left[1] + a[j, 3] - 1
            xtop = top_left[2] + a[j, 2] - 1
            xbot = top_left[2] + a[j, 4] - 1
            rectangular_features[i,j] = boxdiff(img, ytop, xtop, ybot, xbot)        
        end
    end
    rectangular_feature_values = (sum(rectangular_features[:, 2:2:end], 2) - sum(rectangular_features[:, 1:2:end], 2))[:]
end

function haar_features(img::AbstractArray{T, 2}, top_left::Tuple{I, 2}, bottom_right::Tuple{I, 2}, feat::Symbol, coordinates = nothing) where {T <: Union{Real, Color}, I <: Int}
	haar_features(img, [top_left[1], top_left[2]], [bottom_right[1], bottom_right[2]], feat, coordinates)
end

"""
```
coordinates = haar_coordinates(height, width, feat)
```

Returns an array containing the coordinates of the Haar-like features of the specified type.

The caller of the haar_coordinates function specifies a rectangular region in the image by its height and width and a particular
feature type, e.g. two rectangles side by side (:x2). This function then computes the coordinates for the particular rectangle configuration
for all possible locations, and all possible rectangle widths & heights.

Parameters:

 -  height        = Height of the neighbourhood/window where the coordinates are computed
 -  width         = Width of the neighbourhood/window where the coordinates are computed
 -  feat          = A symbol specifying the type of the Haar-like feature to be found

```
    Currently, feat can take 5 values:

    :x2 = Two rectangles along horizontal axis
        +---------------------------------------+
        |                                       |
        | +-----+-----+                         |
        | |     |     |                         |
        | | -1  | +1  |  +---------->           |
        | |     |     |                         |
        | +-----+-----+                         |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :y2 = Two rectangles along vertical axis
        +---------------------------------------+
        |                                       |
        | +------------+                        |
        | |     -1     |                        |
        | +------------+  +---------->          |
        | |     +1     |                        |
        | +------------+                        |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :x3 = Three rectangles along horizontal axis
        +---------------------------------------+
        |                                       |
        | +-----+-----+-----+                   |
        | |     |     |     |                   |
        | | -1  | +1  | -1  |  +---------->     |
        | |     |     |     |                   |
        | +-----+-----+-----+                   |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :y3 = Three rectangles along vertical axis
        +---------------------------------------+
        |                                       |
        | +------------+                        |
        | |     -1     |                        |
        | +------------+                        |
        | |     +1     |  +---------->          |
        | +------------+                        |
        | |     -1     |                        |
        | +------------+                        |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        +---------------------------------------+

    :xy4 = Four rectangles along horizontal and vertical axes
        +---------------------------------------+
        |                                       |
        | +-----+-----+                         |
        | |     |     |                         |
        | | -1  | +1  |                         |
        | |     |     |                         |
        | +-----+-----+  +---------->           |
        | |     |     |                         |
        | | +1  | -1  |                         |
        | |     |     |                         |
        | +-----+-----+                         |
        |                                       |
        |       +                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       |                               |
        |       v                               |
        |                                       |
        |                                       |
        +---------------------------------------+

    The +1 and -1 signs show which rectangles are subtracted and which are added to evaluate the final haar feature.

```

"""
function haar_coordinates(h::Int, w::Int, feat::Symbol)
    rectangular_feature_coordinates = check_feature_type(feat)

    for I = 1:h
        for J = 1:w
            for i = 2:h
                for j = 2:w
                    if feat == :x2 && I + i - 2 <= h && J + 2*j - 3 <= w
                        push!(rectangular_feature_coordinates, (SMatrix{4, 2, Int}(I, J, I + i - 2, J + j - 2, I, J + j - 1, I + i - 2, J + 2*j - 3))')
                    elseif feat == :y2 && I + 2*i - 3 <= h && J + j - 2 <= w
                        push!(rectangular_feature_coordinates, (SMatrix{4, 2, Int}(I, J, I + i - 2, J + j - 2, I + i - 1, J, I + 2*i - 3, J + j - 2))')
                    elseif feat == :x3 && I + i - 2 <= h && J + 3*j - 4 <= w
                        push!(rectangular_feature_coordinates, (SMatrix{4, 3, Int}(I, J, I + i - 2, J + j - 2, I, J + j - 1, I + i - 2, J + 2*j - 3, I, J + 2*j - 2, I + i - 2, J + 3*j - 4))')
                    elseif feat == :y3 && I + 3*i - 4 <= h && J + j  - 2 <= w
                        push!(rectangular_feature_coordinates, (SMatrix{4, 3, Int}(I, J, I + i - 2, J + j - 2, I + i - 1, J, I + 2*i - 3, J + j - 2, I + 2*i - 2, J, I + 3*i - 4, J + j - 2))')
                    elseif feat == :xy4 && I + 2*i - 3 <= h && J + 2*j - 3 <= w
                        push!(rectangular_feature_coordinates, (SMatrix{4, 4, Int}(I, J, I + i - 2, J + j - 2, I, J + j - 1, I + i - 2, J + 2*j - 3, I + i - 1, J + j - 1, I + 2*i - 3, J + 2*j - 3, I + i - 1, J, I + 2*i - 3, J + j - 2))')
                    end
                end
            end
        end
    end

    return rectangular_feature_coordinates
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
        rectangular_feature_coordinates = Vector{SMatrix{2, 4, Int}}()
    elseif feat == :x3 || feat == :y3
        rectangular_feature_coordinates = Vector{SMatrix{3, 4, Int}}()
    elseif feat == :xy4
        rectangular_feature_coordinates = Vector{SMatrix{4, 4, Int}}()
    else
        throw(ArgumentError("The type of the feature must be either :x2, :y2, :x3, :y3 or :xy4."))
    end

    return rectangular_feature_coordinates
end
