using Images, StaticArrays, LinearAlgebra

@testset "haar" begin

    # Basic Haar coordinates tests (:x2)
    test_coordinates = haar_coordinates(2,2,:x2)
    correct_coordinates = [SMatrix{4, 2}(1,1,1,1,1,2,1,2)', SMatrix{4, 2}(2,1,2,1,2,2,2,2)']
    @test test_coordinates == correct_coordinates

    test_coordinates = haar_coordinates(3,2,:x2)
    correct_coordinates = [SMatrix{4, 2}(1,1,1,1,1,2,1,2)', SMatrix{4, 2}(1,1,2,1,1,2,2,2)', SMatrix{4, 2}(2,1,2,1,2,2,2,2)', SMatrix{4, 2}(2,1,3,1,2,2,3,2)', SMatrix{4, 2}(3,1,3,1,3,2,3,2)']
    @test test_coordinates == correct_coordinates

    # Basic Haar coordinates tests (:y2)
    test_coordinates = haar_coordinates(2,2,:y2)
    correct_coordinates = [SMatrix{4, 2}(1,1,1,1,2,1,2,1)', SMatrix{4, 2}(1,2,1,2,2,2,2,2)']
    @test test_coordinates == correct_coordinates

    test_coordinates = haar_coordinates(3,2,:y2)
    correct_coordinates = [SMatrix{4, 2}(1,1,1,1,2,1,2,1)', SMatrix{4, 2}(1,2,1,2,2,2,2,2)', SMatrix{4, 2}(2,1,2,1,3,1,3,1)', SMatrix{4, 2}(2,2,2,2,3,2,3,2)']
    @test test_coordinates == correct_coordinates

    # Basic Haar coordinates tests (:x3)
    test_coordinates = haar_coordinates(3,2,:x3)
    @test length(test_coordinates) == 0

    test_coordinates = haar_coordinates(3,3,:x3)
    correct_coordinates = [SMatrix{4, 3}(1,1,1,1,1,2,1,2,1,3,1,3)', SMatrix{4, 3}(1,1,2,1,1,2,2,2,1,3,2,3)', SMatrix{4, 3}(2,1,2,1,2,2,2,2,2,3,2,3)', SMatrix{4, 3}(2,1,3,1,2,2,3,2,2,3,3,3)', SMatrix{4, 3}(3,1,3,1,3,2,3,2,3,3,3,3)']
    @test test_coordinates == correct_coordinates

    # Basic Haar coordinates tests (:y3)
    test_coordinates = haar_coordinates(2,3,:y3)
    @test length(test_coordinates) == 0

    test_coordinates = haar_coordinates(3,3,:y3)
    correct_coordinates = [SMatrix{4, 3}(1,1,1,1,2,1,2,1,3,1,3,1)', SMatrix{4, 3}(1,1,1,2,2,1,2,2,3,1,3,2)', SMatrix{4, 3}(1,2,1,2,2,2,2,2,3,2,3,2)', SMatrix{4, 3}(1,2,1,3,2,2,2,3,3,2,3,3)', SMatrix{4, 3}(1,3,1,3,2,3,2,3,3,3,3,3)']
    @test test_coordinates == correct_coordinates

    # Basic Haar coordinates tests (:xy4)
    test_coordinates = haar_coordinates(2,2,:xy4)
    correct_coordinates = [SMatrix{4, 4}(1,1,1,1,1,2,1,2,2,2,2,2,2,1,2,1)']
    @test test_coordinates == correct_coordinates

    test_coordinates = haar_coordinates(3,2,:xy4)
    correct_coordinates = [SMatrix{4, 4}(1,1,1,1,1,2,1,2,2,2,2,2,2,1,2,1)', SMatrix{4, 4}(2,1,2,1,2,2,2,2,3,2,3,2,3,1,3,1)']
    @test test_coordinates == correct_coordinates

    test_coordinates = haar_coordinates(3,3,:xy4)
    correct_coordinates = [SMatrix{4, 4}(1,1,1,1,1,2,1,2,2,2,2,2,2,1,2,1)', SMatrix{4, 4}(1,2,1,2,1,3,1,3,2,3,2,3,2,2,2,2)', SMatrix{4, 4}(2,1,2,1,2,2,2,2,3,2,3,2,3,1,3,1)', SMatrix{4, 4}(2,2,2,2,2,3,2,3,3,3,3,3,3,2,3,2)']
    @test test_coordinates == correct_coordinates

    # Simple Image haar_features

    img = Matrix(Diagonal([1,1,1,1,1]))
    int_img = IntegralArray(img)

    test_features = haar_features(int_img, [2,2], [4,4], :x2)
    correct_features = [-1,0,0,-1,1,1,-1,0,0,1]
    @test test_features == correct_features

    test_features = haar_features(int_img, [1,1], [4,4], :xy4)
    correct_features = [-2,0,0,-4,1,-2,0,0,1,-2,-2,1,0,0,1,-2]
    @test test_features == correct_features

    # Simple Image haar_features with coordinates given

    coordinates = rand(Int,3,3,4)
    coordinates[1,1,:] = [1,3,1,3]
    coordinates[1,2,:] = [1,4,1,4]
    coordinates[1,3,:] = [1,5,1,5]
    coordinates[2,1,:] = [3,2,3,2]
    coordinates[2,2,:] = [3,3,3,3]
    coordinates[2,3,:] = [3,4,3,4]
    coordinates[3,1,:] = [5,1,5,1]
    coordinates[3,2,:] = [5,2,5,2]
    coordinates[3,3,:] = [5,3,5,3]
    test_features = haar_features(int_img, [1,1], [5,5], :x3, coordinates)
    correct_features = [0,1,0]
    @test test_features == correct_features

    # Tests for error checks

    @test_throws ArgumentError haar_features(int_img, [1,1,1], [5,5], :x3)
    @test_throws ArgumentError haar_features(int_img, [7,1], [5,5], :x3)
    @test_throws ArgumentError haar_features(int_img, [1,7], [5,5], :x3)
    @test_throws ArgumentError haar_features(int_img, [1,1], [5,5], :x4)

    #No feature coordinates found case
    @test length(haar_features(int_img, [1,1], [1,5], :x2)) == 0
end
