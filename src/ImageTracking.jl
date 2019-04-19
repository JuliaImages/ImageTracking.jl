__precompile__()

module ImageTracking

using Images
using ImageFiltering
using Interpolations
using StaticArrays
using LinearAlgebra
using CoordinateTransformations
using Statistics

abstract type VisualizationMethod end
struct ColorBased <: VisualizationMethod end

abstract type AbstractFlowError end
struct EndpointError <: AbstractFlowError end
struct AngularError <: AbstractFlowError end

abstract type AbstractCoordinateConvention end
struct RasterConvention <: AbstractCoordinateConvention end
struct CartesianConvention <: AbstractCoordinateConvention end

include("core.jl")
include("optical_flow.jl")
include("haar.jl")
include("utility.jl")
include("tracker/tracker.jl")

export

	# main functions
    optical_flow,
    optical_flow!,

	# other functions
	haar_coordinates,
	haar_features,

	# other functions
	ColorBased,
	EndpointError,
	AngularError,
	polynomial_expansion,
	RasterConvention,
	CartesianConvention,
	visualize_flow,
	read_flow_file,
	evaluate_flow_error,
	calculate_statistics,

	# optical flow algorithms
	LucasKanade,
	Farneback,

    # types that select implementation
    ConvolutionImplementation,
    MatrixImplementation,

    # tracking algorithms
    BoxROI,
    CurrentSampler,
    TrackerBoosting,
    initialize!,
    update!

end
