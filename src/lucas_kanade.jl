"""
    LK(Args...)

A differential method for optical flow estimation developed by Bruce D. Lucas 
and Takeo Kanade. It assumes that the flow is essentially constant in a local 
neighbourhood of the pixel under consideration, and solves the basic optical flow 
equations for all the pixels in that neighbourhood, by the least squares criterion.

The different arguments are:

 -  prev_points       =  Vector of Coordinates for which the flow needs to be found
 -  next_points       =  Vector of Coordinates containing initial estimates of new positions of 
                         input features in next image
 -  window_size       =  Size of the search window at each pyramid level; the total size of the 
                         window used is 2*window_size + 1
 -  max_level         =  0-based maximal pyramid level number; if set to 0, pyramids are not used 
                         (single level), if set to 1, two levels are used, and so on
 -  estimate_flag     =  0 -> Use next_points as initial estimate (Default Value)
                         1 -> Copy prev_points to next_points and use as estimate
 -  term_condition    =  The termination criteria of the iterative search algorithm i.e the number of iterations
 -  min_eigen_thresh  =  The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of optical 
                         flow equations, divided by number of pixels in a window; if this value is less than 
                         min_eigen_thresh, then a corresponding feature is filtered out and its flow is not processed

## References

B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision," 
DARPA Image Understanding Workshop, pp 121-130, 1981.

J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas kanadefeature tracker description of the 
algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.
"""

struct LK{} <: OpticalFlowAlgo
    prev_points::Array{Coordinate{Int64}, 1}
    next_points::Array{Coordinate{Float64}, 1}
    window_size::Int64
    max_level::Int64
    estimate_flag::Bool
    term_condition::Int64
    min_eigen_thresh::Float64
end
