using Images, TestImages, StaticArrays, OffsetArrays, FileIO, JLD2

# Testing constants
test_image = "mandrill"
number_test_pts = 500
difference = 0.3
max_error_points_percentage = 10
max_allowed_error = 0.1
max_lost_points_percentage = 40

function test_lk(number_test_pts::Int, flow::Array{SVector{2, Float64}, 1}, x_flow::Float64, y_flow::Float64, status::BitArray{1}, err::Array{Float64, 1}, difference::Float64)
    #Testing parameters
    error_pts = 0
    max_err = 0
    total_err = 0
    lost_points = 0

    for i = 1:number_test_pts
        if status[i]
            if abs(flow[i][2] - x_flow) > difference || abs(flow[i][1] - y_flow) > difference
                error_pts += 1
            end
            if err[i] > max_err
                max_err = err[i]
            end
            total_err += err[i]
        else
            lost_points += 1
        end
    end
    return error_pts, max_err, total_err, lost_points
end

@testset "lucas-kanade" begin

    #Basic translations (Horizontal)
    img1 = Gray{Float64}.(testimage(test_image))
    img2 = similar(img1)
    for i = 1:size(img1)[1]
        for j = 4:size(img1)[2]
            img2[i,j] = img1[i,j-3]
        end
    end

    corners = imcorner(img1, method=shi_tomasi)
    y, x = findn(corners)
    a = map((yi, xi) -> SVector{2}(yi, xi), y, x)

    srand(9876)
    pts = rand(a, (number_test_pts,))

    flow, status, err = optical_flow(img1, img2, LK(pts, [SVector{2}(0.0,0.0)], 11, 4, false, 20))

    error_pts, max_err, total_err, lost_points = test_lk(number_test_pts, flow, 3.0, 0.0, status, err, difference)

    println("Horizontal Motion")
    println("Error Points Percentage = ", (error_pts/length(pts))*100)
    @test ((error_pts/length(pts))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
    @test max_err < max_allowed_error
    println("Lost Points Percentage = ", (lost_points/length(pts))*100)
    @test ((lost_points/length(pts))*100) < max_lost_points_percentage


    # Basic translations (Vertical)
    img2 = OffsetArray(img1, 3, 0)

    flow, status, err = optical_flow(img1, img2, LK(pts, [SVector{2}(0.0,0.0)], 11, 4, false, 20))

    error_pts, max_err, total_err, lost_points = test_lk(number_test_pts, flow, 0.0, 3.0, status, err, difference)

    println("Vertical Motion")
    println("Error Points Percentage = ", (error_pts/length(pts))*100)
    @test ((error_pts/length(pts))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
    @test max_err < max_allowed_error
    println("Lost Points Percentage = ", (lost_points/length(pts))*100)
    @test ((lost_points/length(pts))*100) < max_lost_points_percentage


    # Basic translations (Both)
    img2 = OffsetArray(img1, 3, 1)

    flow, status, err = optical_flow(img1, img2, LK(pts, [SVector{2}(0.0,0.0)], 11, 4, false, 20))

    error_pts, max_err, total_err, lost_points = test_lk(number_test_pts, flow, 1.0, 3.0, status, err, difference)

    println("Combined Motion")
    println("Error Points Percentage = ", (error_pts/length(pts))*100)
    @test ((error_pts/length(pts))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
    @test max_err < max_allowed_error
    println("Lost Points Percentage = ", (lost_points/length(pts))*100)
    @test ((lost_points/length(pts))*100) < max_lost_points_percentage


    #Yosemite Sequence
    img1 = load("test_data/yosemite/images/img1.tif")
    img2 = load("test_data/yosemite/images/img2.tif")

    corners = imcorner(img1, 0.0025, method=shi_tomasi)
    y, x = findn(corners)
    a = map((yi, xi) -> SVector{2}(yi, xi), y, x)

    flow, status, err = optical_flow(img1, img2, LK(a, [SVector{2}(0.0,0.0)], 25, 4, false, 20))

    correct_flow = load("test_data/yosemite/flow/yosemite_correct_flow.jld2", "yosemite_correct_flow")

    error_pts = 0
    lost_points = 0
    for i = 1:length(a)
        if !status[i]
            lost_points += 1
        else
            if abs(correct_flow[a[i]..., 1] - flow[i][1]) > 1 || abs(correct_flow[a[i]..., 2] - flow[i][2]) > 1
                error_pts += 1
            end
        end
    end

    @test ((error_pts/length(a))*100) < 25
    @test ((lost_points/length(a))*100) < 10
end
