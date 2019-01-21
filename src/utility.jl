
function visualize_optical_flow_hsv(flow::Array{SVector{2, Float64}, 2})
 
    # Convert from (row,column) to (x,y) convention.
    map!(x-> SVector(last(x),first(x)), flow, flow)

    # Display optical flow as an image, with hue encoding the orientation and
    # saturation encoding the relative magnitude.
    max_norm = maximum(map(norm,flow))
    normalised_flow = map(PolarFromCartesian(),flow / max_norm)
    hsv = zeros(HSV{Float32},size(flow))

    for i in eachindex(flow)
        hsv[i] = HSV((normalised_flow[i].θ + pi) * 180 / pi, normalised_flow[i].r, 1)
    end

    return hsv
    
end
