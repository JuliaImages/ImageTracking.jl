mutable struct SingleHaarFeature
    #Uninitialized
    rect_type::Symbol
    num_of_areas::Int
    areas::MMatrix{4, N, Int} where N
    weights::MVector{N, Float64} where N

    SingleHaarFeature() = new()
end

mutable struct HaarTrackerFeatures{I <: Int}
    #Initialized
    num_of_features::I
    patch_size::MVector{2, I}

    #Uinitialized
    features::Vector{SingleHaarFeature}
    responses::Array{T, 2} where T

    HaarTrackerFeatures(num_of_features::I, patch_size::MVector{2, I}) where I <: Int = new{I}(
                        num_of_features, patch_size)
end

function initialize_feature_haar(patch_size::MVector{2, Int})
    selected = false

    haar_feature = SingleHaarFeature()

    while !selected
        location = SVector{2}(ceil(Int, rand()*patch_size[1]), ceil(Int, rand()*patch_size[2]))
        dimensions = SVector{2}(round(Int, 0.75*rand()*patch_size[1]), round(Int, 0.75*rand()*patch_size[2]))

        case = rand([1, 2, 3, 4, 5])

        if case == 1
            #CHECK: Added -1 here
            if location[1] + dimensions[1] - 1 > patch_size[1] || location[2] + 2*dimensions[2] - 1 > patch_size[2] || 2*dimensions[1]*dimensions[2] < 9
                continue
            end

            haar_feature.rect_type = :x2
            haar_feature.num_of_areas = 2
            haar_feature.weights = MVector{2}(1.0, -1.0)
            haar_feature.areas = MMatrix{4,2}(location[1], location[2], location[1] + dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1], location[2] + dimensions[2], location[1] + dimensions[1] - 1, location[2] + 2*dimensions[2] - 1)

            selected = true
        elseif case == 2
            if location[1] + 2*dimensions[1] - 1 > patch_size[1] || location[2] + dimensions[2] - 1 > patch_size[2] || 2*dimensions[1]*dimensions[2] < 9
                continue
            end

            haar_feature.rect_type = :y2
            haar_feature.num_of_areas = 2
            haar_feature.weights = MVector{2}(1.0, -1.0)
            haar_feature.areas = MMatrix{4,2}(location[1], location[2], location[1] + dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1] + dimensions[1], location[2], location[1] + 2*dimensions[1] - 1, location[2] + dimensions[2] - 1)

            selected = true
        elseif case == 3
            if location[1] + dimensions[1] - 1 > patch_size[1] || location[2] + 4*dimensions[2] - 1 > patch_size[2] || 4*dimensions[1]*dimensions[2] < 9
                continue
            end

            haar_feature.rect_type = :x3
            haar_feature.num_of_areas = 3
            haar_feature.weights = MVector{3}(1.0, -2.0, 1.0)
            haar_feature.areas = MMatrix{4,3}(location[1], location[2], location[1] + dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1], location[2] + dimensions[2], location[1] + dimensions[1] - 1, location[2] + 3*dimensions[2] - 1,
                                              location[1], location[2] + 3*dimensions[2], location[1] + dimensions[1] - 1, location[2] + 4*dimensions[2] - 1)

            selected = true
        elseif case == 4
            if location[1] + 4*dimensions[1] - 1 > patch_size[1] || location[2] + dimensions[2] - 1 > patch_size[2] || 4*dimensions[1]*dimensions[2] < 9
                continue
            end

            haar_feature.rect_type = :y3
            haar_feature.num_of_areas = 3
            haar_feature.weights = MVector{3}(1.0, -2.0, 1.0)
            haar_feature.areas = MMatrix{4,3}(location[1], location[2], location[1] + dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1] + dimensions[1], location[2], location[1] + 3*dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1] + 3*dimensions[1], location[2], location[1] + 4*dimensions[1] - 1, location[2] + dimensions[2] - 1)

            selected = true
        else
            if location[1] + 2*dimensions[1] - 1 > patch_size[1] || location[2] + 2*dimensions[2] - 1 > patch_size[2] || 2*2*dimensions[1]*dimensions[2] < 9
                continue
            end

            haar_feature.rect_type = :xy4
            haar_feature.num_of_areas = 3
            haar_feature.weights = MVector{4}(1.0, -1.0, -1.0, 1.0)
            haar_feature.areas = MMatrix{4,4}(location[1], location[2], location[1] + dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1], location[2] + dimensions[2], location[1] + dimensions[1] - 1, location[2] + 2*dimensions[2] - 1,
                                              location[1] + dimensions[1], location[2], location[1] + 2*dimensions[1] - 1, location[2] + dimensions[2] - 1,
                                              location[1] + dimensions[1], location[2] + dimensions[2], location[1] + 2*dimensions[1] - 1, location[2] + 2*dimensions[2] - 1)

            selected = true
        end
    end

    return haar_feature
end

function evaluate_haar_feature(feature::SingleHaarFeature, image::Array{T, 2}) where T
    result = 0.0
    for i = 1:feature.num_of_areas
        result += feature.weights[i]*boxdiff(Float64.(image), feature.areas[1,i]:feature.areas[3,i], feature.areas[2,i]:feature.areas[4,i])
    end
    return result
end

#HaarTrackerFeatures

function generate_features(features::HaarTrackerFeatures{}, num_of_features::Int)
    features.features = Vector{SingleHaarFeature}()
    for i = 1:num_of_features
        push!(features.features, initialize_feature_haar(features.patch_size))
    end
end

function swap_feature(features::HaarTrackerFeatures{}, index::Int, feature::SingleHaarFeature)
    features.features[index] = feature
end

function swap_feature(features::HaarTrackerFeatures{}, source_index::Int, target_index::Int)
    temp_feature = features.features[source_index]
    features.features[source_index] = features.features[target_index]
    features.features[target_index] = temp_feature
end

function extract_selected(features::HaarTrackerFeatures{}, sel_features::MVector{N1, Int}, samples::MVector{N2, Tuple{Array{T, 2}, MVector{2, Int}}}) where N1 where N2 where T
    response = Array{Float64}(length(samples), features.num_of_features)

    for i = 1:length(samples)
        for j = 1:length(sel_features)
            response[i, sel_features[j]] = Float64(evaluate_haar_feature(features.features[sel_features[j]], samples[i][1]))
        end
    end

    return response
end

function extraction(features::HaarTrackerFeatures{}, samples::MVector{N, Tuple{Array{T, 2}, MVector{2, Int}}}) where N where T
    response = Array{Float64}(length(samples), features.num_of_features)

    for j = 1:features.num_of_features
        for i = length(samples)
            response[i,j] = evaluate_haar_feature(features.features[j], samples[i][1])
        end
    end

    features.responses = response
end
