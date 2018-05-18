"""
	`Coordinate` type contains two fields x and y representing the 
	Cartesian coordinates.
"""

mutable struct Coordinate{T <: Union{Int64,Float64}}
    x::T
    y::T
end

import Base.+
import Base.*

+(a::Coordinate,b::Coordinate) = Coordinate(a.x+b.x,a.y+b.y)
+(a::Array{UnitRange{Int64},1},b::Coordinate{Float64}) = [a[1]+b.y,a[2]+b.x]
+(a::Array{StepRangeLen{Float64,Base.TwicePrecision{Float64}},1},b::Coordinate{Float64}) = [a[1]+b.y,a[2]+b.x]
+(a::Array{ColorTypes.Gray{Float64},1},b::Coordinate{Float64}) = Coordinate{Float64}(a[2]+b.x,a[1]+b.y)

*(a::Int64,b::Coordinate{Float64}) = Coordinate(a*b.x,a*b.y)
*(a::Float64,b::Coordinate{Float64}) = Coordinate(a*b.x,a*b.y)
