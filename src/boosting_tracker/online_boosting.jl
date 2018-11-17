mutable struct WeakClassifier
    μ::Float64
    σ::Float64
    positive_samples_dist::MVector{6, Float64}
    negative_samples_dist::MVector{6, Float64}
    threshold::Float64
    parity::Int
end

mutable struct BaseClassifier
    weak_classifiers::Vector{WeakClassifier}
    reference_weak_classifier::Bool
    num_of_weak_classifiers::Int
    selected_classifier::Int
    new_weak_classifier_index::Int
    w_correct::MVector{N, Float64} where N
    w_wrong::MVector{N, Float64} where N
    initial_iterations::Int
end

mutable struct Detector
    #Initialized
    size_of_confidences::Int
    num_of_detections::Int
    size_of_detections::Int
    best_detection_index::Int
    max_confidence::Float64

    #Uinitialized
    confidences::MVector{N, Float64} where N
    detection_indices::MVector{N, Int} where N
    confidence_matrix::MMatrix{N1, N2, Float64} where N1 where N2

    Detector(size_of_confidences::Int = 0, num_of_detections::Int = 0, size_of_detections::Int = 0, best_detection_index::Int = 0, max_confidence::Float64 = -Inf) = new(
             size_of_confidences, num_of_detections, size_of_detections, best_detection_index, max_confidence)
end

mutable struct StrongClassifier
    #Initialized
    num_of_base_classifiers::Int
    num_of_weak_classifiers::Int
    initial_iterations::Int
    α::MVector{N, Float64} where N
    patch_size::MVector{2, Int}
    feature_exchange::Bool
    error_mask::MVector{N, Bool} where N
    errors::MVector{N, Float64} where N
    sum_of_errors::MVector{N, Float64} where N
    detector::Detector
    ROI::MVector{4, Int}

    #Uinitialized
    base_classifiers::MVector{N, BaseClassifier} where N
    replaced_classifier::Int
    swapped_classifier::Int

    StrongClassifier(num_of_base_classifiers::Int, num_of_weak_classifiers::Int, initial_iterations::Int, α::MVector{N1, Float64}, patch_size::MVector{2, Int}, feature_exchange::Bool, error_mask::MVector{N2, Bool}, errors::MVector{N2, Float64}, sum_of_errors::MVector{N2, Float64}, detector::Detector, ROI::MVector{4, Int}) where N1 where N2 = new(
                     num_of_base_classifiers, num_of_weak_classifiers, initial_iterations, α, patch_size, feature_exchange, error_mask, errors, sum_of_errors, detector, ROI)
end

#StrongClassifier

function initialize_strong_classifier(num_of_base_classifiers::Int, num_of_weak_classifiers::Int, initial_iterations::Int, patch_size::MVector{2, Int}, feature_exchange::Bool, ROI::MVector{4, Int})
    α = MVector{num_of_base_classifiers}(zeros(num_of_base_classifiers,))
    error_mask = MVector{num_of_weak_classifiers + initial_iterations, Bool}()
    errors = MVector{num_of_weak_classifiers + initial_iterations, Float64}()
    sum_of_errors = MVector{num_of_weak_classifiers + initial_iterations, Float64}()
    detector = Detector()
    StrongClassifier(num_of_base_classifiers, num_of_weak_classifiers, initial_iterations, α, patch_size, feature_exchange, error_mask, errors, sum_of_errors, detector, ROI)
end

function init_base_classifiers(sc::StrongClassifier)
    tempBaseClassifier = [initialize_base_classifier(sc.num_of_weak_classifiers, sc.initial_iterations)]
    for i = 2:sc.num_of_base_classifiers
        push!(tempBaseClassifier, initialize_base_classifier(sc.num_of_weak_classifiers, sc.initial_iterations, tempBaseClassifier[1].weak_classifiers))
    end
    sc.base_classifiers = MVector{sc.num_of_base_classifiers}(tempBaseClassifier)
end

function classify(sc::StrongClassifier, images::MVector{N, Array{T, 2}}, sample_ROI::MVector{4, Int}) where T where N
    sc.ROI = sample_ROI
    index = 0
    confidences = 0.0

    num_patches = length(images)
    sc.detector.size_of_confidences = num_patches
    sc.detector.confidences = MVector{num_patches, Float64}()

    sc.detector.num_of_detections = 0
    sc.detector.best_detection_index = -1
    sc.detector.max_confidence = -Inf

    step_row = max(1, floor(Int, 0.01*sc.patch_size[1] + 0.5))
    step_col = max(1, floor(Int, 0.01*sc.patch_size[2] + 0.5))

    height = floor(Int,(sc.ROI[3] - sc.ROI[1] - sc.patch_size[1])/step_row)
    width = floor(Int,(sc.ROI[4] - sc.ROI[2] - sc.patch_size[2])/step_col)

    sc.detector.confidence_matrix = MMatrix{height, width, Float64}()

    current_patch = 1
    for i = 1:width
        for j = 1:height
            sc.detector.confidences[current_patch] = evaluate(sc, images[current_patch])

            sc.detector.confidence_matrix[j,i] = sc.detector.confidences[current_patch]
            current_patch += 1
        end
    end

    sc.detector.confidence_matrix = ImageFiltering.imfilter(sc.detector.confidence_matrix, centered(Kernel.gaussian(0.5)[-1:1,-1:1]))

    min_val = minimum(sc.detector.confidence_matrix)
    max_val = maximum(sc.detector.confidence_matrix)

    current_patch = 1
    for i = 1:width
        for j = 1:height
            if sc.detector.confidences[current_patch] > sc.detector.max_confidence
                sc.detector.max_confidence = sc.detector.confidences[current_patch]
                sc.detector.best_detection_index = current_patch
            end

            if sc.detector.confidences[current_patch] > 0.0
                sc.detector.num_of_detections += 1
            end

            current_patch += 1
        end
    end

    sc.detector.size_of_detections = sc.detector.num_of_detections
    sc.detector.detection_indices = MVector{sc.detector.num_of_detections, Int}()

    current_detection = 1
    for i = 1:num_patches
        if sc.detector.confidences[i] > 0.0
            sc.detector.detection_indices[current_detection] = i
            current_detection += 1
        end
    end

    if sc.detector.num_of_detections <= 0
        warn("No detections!")
        confidences = 0
        return index, confidences
    end

    index = sc.detector.best_detection_index
    confidences = sc.detector.max_confidence
    return index, confidences
end

function update(sc::StrongClassifier, image::AbstractArray{T, 2}, target::Int, importance::Float64 = 1.0) where T
    sc.error_mask .= false
    sc.errors .= 0.0
    sc.sum_of_errors .= 0.0

    train_classifier(sc.base_classifiers[1], image, target, importance, sc.error_mask)
    for i = 1:sc.num_of_base_classifiers
        selected_classifier = select_best_classifier(sc.base_classifiers[i], sc.errors, sc.error_mask, importance)

        if sc.errors[selected_classifier] >= 0.5
            sc.α[i] = 0
        else
            sc.α[i] = log((1.0 - sc.errors[selected_classifier])/sc.errors[selected_classifier])
        end

        if sc.error_mask[selected_classifier]
            importance *= sqrt((1.0 - sc.errors[selected_classifier])/sc.errors[selected_classifier])
        else
            importance *= sqrt(sc.errors[selected_classifier]/(1.0 - sc.errors[selected_classifier]))
        end

        for j = i:sc.num_of_weak_classifiers
            if sc.errors[j] != Inf && sc.sum_of_errors[j] >= 0
                sc.sum_of_errors[j] += sc.errors[j]
            end
        end

        sc.sum_of_errors[selected_classifier] = -1
        sc.errors[selected_classifier] = Inf
    end

    if sc.feature_exchange
        sc.replaced_classifier = compute_replacement(sc.base_classifiers[1], sc.sum_of_errors)
        sc.swapped_classifier = sc.base_classifiers[1].new_weak_classifier_index
    end

    return true
end

function replace_weak_classifier(sc::StrongClassifier, index::Int)
    if sc.feature_exchange && index > 0
        replace_weak_classifier(sc.base_classifiers[1], index)

        source_index = sc.base_classifiers[1].new_weak_classifier_index
        target_index = index

        for i = 2:sc.num_of_base_classifiers
            assert(target_index > 0)
            assert(target_index != sc.base_classifiers[i].selected_classifier)
            assert(target_index <= sc.base_classifiers[i].num_of_weak_classifiers)

            sc.base_classifiers[i].w_wrong[target_index] = sc.base_classifiers[i].w_wrong[source_index]
            sc.base_classifiers[i].w_wrong[source_index] = 1.0
            sc.base_classifiers[i].w_correct[target_index] = sc.base_classifiers[i].w_correct[source_index]
            sc.base_classifiers[i].w_correct[source_index] = 1.0
        end
    end
end

function get_selected_weak_classifier(sc::StrongClassifier)
    return map(i -> i.selected_classifier, sc.base_classifiers)
end

function evaluate(sc::StrongClassifier, response::AbstractArray{T, 2}) where T
    result = 0.0
    for i = 1:sc.num_of_base_classifiers
        result += evaluate(sc.base_classifiers[i], response)*sc.α[i]
    end
    return result
end

#BaseClassifier

function initialize_base_classifier(num_of_weak_classifiers::Int, initial_iterations::Int)
    weak_classifiers = Vector{WeakClassifier}()
    for i = 1:num_of_weak_classifiers+initial_iterations
        push!(weak_classifiers, WeakClassifier(0.0, 1.0, MVector{6, Float64}(0.0, 1.0, 1000.0, 0.01, 1000.0, 0.01), MVector{6, Float64}(0.0, 1.0, 1000.0, 0.01, 1000.0, 0.01), 0.0, 0.0))
    end
    w_correct = MVector{num_of_weak_classifiers+initial_iterations}(ones(num_of_weak_classifiers+initial_iterations,))
    w_wrong = MVector{num_of_weak_classifiers+initial_iterations}(ones(num_of_weak_classifiers+initial_iterations,))
    return BaseClassifier(weak_classifiers, false, num_of_weak_classifiers, 1, num_of_weak_classifiers, w_correct, w_wrong, initial_iterations)
end

function initialize_base_classifier(num_of_weak_classifiers::Int, initial_iterations::Int, weak_classifiers::Vector{WeakClassifier})
    w_correct = MVector{num_of_weak_classifiers+initial_iterations}(ones(num_of_weak_classifiers+initial_iterations,))
    w_wrong = MVector{num_of_weak_classifiers+initial_iterations}(ones(num_of_weak_classifiers+initial_iterations,))
    return BaseClassifier(weak_classifiers, true, num_of_weak_classifiers, 1, num_of_weak_classifiers, w_correct, w_wrong, initial_iterations)
end

function evaluate(bc::BaseClassifier, image::AbstractArray{T, 2}) where T
    #CHECK: Made it 1 from : in last argument
    wc = bc.weak_classifiers[bc.selected_classifier]
    value = image[bc.selected_classifier,1]
    return (wc.parity*(value - wc.threshold) > 0 ? 1 : -1)
end

function train_classifier(bc::BaseClassifier, image::AbstractArray{T, 2}, target::Int, importance::Float64, error_mask::MVector{N, Bool}) where T where N
    A = 1.0
    K = 0
    K_Max = 10
    while true
        A *= rand()
        if K > K_Max || A < exp(-importance)
            break
        end
        K += 1
    end

    for i = 1:K+1
        for j = 1:bc.num_of_weak_classifiers+bc.initial_iterations
            error_mask[j] = update(bc.weak_classifiers[j], Float64.(image[j,1]), target)
        end
    end
end

function select_best_classifier(bc::BaseClassifier, errors::MVector{N, Float64}, errorMask::MVector{N, Bool}, importance::Float64) where N
    minError = Inf
    temp_selected_classifier = bc.selected_classifier
    for i = 1:bc.num_of_weak_classifiers+bc.initial_iterations
        if errorMask[i]
            bc.w_wrong[i] += importance
        else
            bc.w_correct[i] += importance
        end

        if errors[i] == Inf
            continue
        end

        errors[i] = bc.w_wrong[i]/(bc.w_correct[i] + bc.w_wrong[i])

        if i < bc.num_of_weak_classifiers && errors[i] < minError
            minError = errors[i]
            temp_selected_classifier = i
        end
    end

    bc.selected_classifier = temp_selected_classifier
    return bc.selected_classifier
end

function replace_weak_classifier(bc::BaseClassifier, index::Int)
    bc.weak_classifiers[index] = bc.weak_classifiers[bc.new_weak_classifier_index]
    bc.w_wrong[index] = bc.w_wrong[bc.new_weak_classifier_index]
    bc.w_correct[index] = bc.w_correct[bc.new_weak_classifier_index]

    bc.w_wrong[bc.new_weak_classifier_index] = 1
    bc.w_correct[bc.new_weak_classifier_index] = 1
    bc.weak_classifiers[bc.new_weak_classifier_index] = WeakClassifier(0.0, 1.0, MVector{6, Float64}(0.0, 1.0, 1000.0, 0.01, 1000.0, 0.01), MVector{6, Float64}(0.0, 1.0, 1000.0, 0.01, 1000.0, 0.01), 0.0, 0.0)
end

function compute_replacement(bc::BaseClassifier, errors::MVector{N, Float64}) where N
    maxError = 0.0
    #Taken index 0 rather than -1
    index = 0

    for i = bc.num_of_weak_classifiers:-1:1
        if errors[i] > maxError
            maxError = errors[i]
            index = i
        end
    end

    assert(index > 0)
    assert(index != bc.selected_classifier)

    bc.new_weak_classifier_index += 1
    if bc.new_weak_classifier_index == bc.num_of_weak_classifiers + bc.initial_iterations
        bc.new_weak_classifier_index = bc.num_of_weak_classifiers
    end

    if maxError > errors[bc.new_weak_classifier_index]
        return index
    else
        return 0
    end
end

#WeakClassifier

function update(wc::WeakClassifier, value::Float64, target::Int)
    if target == 1
        update(wc.positive_samples_dist, value)
    else
        update(wc.negative_samples_dist, value)
    end

    wc.threshold = (wc.positive_samples_dist[2] + wc.negative_samples_dist[2])/2
    wc.parity = (wc.positive_samples_dist[2] > wc.negative_samples_dist[2] ? 1 : -1)

    return ((wc.parity*(value - wc.threshold) > 0 ? 1 : -1) != target)
end

#Misc

function update(gauss_dist::MVector{6, Float64}, value::Float64)
    min_factor = 0.001

    K = gauss_dist[3]/(gauss_dist[3] + gauss_dist[5])
    if K < min_factor
        K = min_factor
    end

    gauss_dist[1] = K*value + (1 - K)*gauss_dist[1]
    gauss_dist[3] = (gauss_dist[3]*gauss_dist[5])/(gauss_dist[3] + gauss_dist[5])

    K = gauss_dist[4]/(gauss_dist[4] + gauss_dist[6])
    if K < min_factor
        K = min_factor
    end

    temp_σ = K*(gauss_dist[1] - value)^2 + (1 - K)*gauss_dist[2]^2
    gauss_dist[4] = (gauss_dist[4]*gauss_dist[6])/(gauss_dist[4] + gauss_dist[6])

    gauss_dist[2] = sqrt(temp_σ)
    if gauss_dist[2] <= 1.0
      gauss_dist[2] = 1.0
    end
end
