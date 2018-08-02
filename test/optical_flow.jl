using Images, TestImages, StaticArrays, OffsetArrays

#Lucas-Kanade Optical Flow tests

    # Testing constants
    test_image = "mandrill"
    number_test_pts = 500
    difference = 0.3
    max_error_points_percentage = 10
    max_allowed_error = 0.1
    max_lost_points_percentage = 40

function test_lk(number_test_pts::Int64, flow::Array{SVector{2, Float64}, 1}, x_flow::Float64, y_flow::Float64, status::BitArray{1}, err::Array{Float64, 1}, difference::Float64)
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

    println("Lucas-Kanade TestSet!")

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
end

#Farneback Optical Flow tests

    # Testing constants
    test_image = "mandrill"
    difference = 0.3
    max_error_points_percentage = 7.5

function test_fb(dims::SVector{2, Int}, flow::Array{SVector{2, Float64}, 2}, x_flow::Float64, y_flow::Float64, difference::Float64)
    #Testing parameters
    error_pts = 0
    max_err = 0

    for i = 1:dims[2]
        for j = 1:dims[1]
            if abs(flow[j,i][1] - x_flow) > difference || abs(flow[j,i][2] - y_flow) > difference
                    error_pts += 1
            end
            err = sqrt((abs(flow[j,i][1] - x_flow))^2 + (abs(flow[j,i][2] - y_flow))^2)
            if err > max_err
                max_err = err
            end
        end
    end
    return error_pts, max_err
end

@testset "farneback" begin

    println("Farneback TestSet!")

    #Basic translations (Horizontal)
    img1 = Gray{Float64}.(testimage(test_image))
    img2 = similar(img1)
    for i = 1:size(img1)[1]
        for j = 4:size(img1)[2]
            img2[i,j] = img1[i,j-3]
        end
    end

    flow = optical_flow(img1, img2, Farneback(rand(SVector{2,Float64},2,2), 7, 39, 6.0, 11, 1.5, false, true))

    error_pts, max_err = test_fb(SVector{2}(size(img1)), flow, 3.0, 0.0, difference)

    println("Horizontal Motion")
    println("Error Points Percentage = ", (error_pts/(size(img1)[1]*size(img1)[2]))*100)
    @test ((error_pts/(size(img1)[1]*size(img1)[2]))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)


    # Basic translations (Vertical)
    img2 = similar(img1)
    for i = 4:size(img1)[1]
        for j = 1:size(img1)[2]
            img2[i,j] = img1[i-3,j]
        end
    end

    flow = optical_flow(img1, img2, Farneback(rand(SVector{2,Float64},2,2), 7, 39, 6.0, 11, 1.5, false, true))

    error_pts, max_err = test_fb(SVector{2}(size(img1)), flow, 0.0, 3.0, difference)

    println("Horizontal Motion")
    println("Error Points Percentage = ", (error_pts/(size(img1)[1]*size(img1)[2]))*100)
    @test ((error_pts/(size(img1)[1]*size(img1)[2]))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)


    # Basic translations (Both)
    img2 = similar(img1)
    for i = 4:size(img1)[1]
        for j = 2:size(img1)[2]
            img2[i,j] = img1[i-3,j-1]
        end
    end

    flow = optical_flow(img1, img2, Farneback(rand(SVector{2,Float64},2,2), 7, 39, 6.0, 11, 1.5, false, true))

    error_pts, max_err = test_fb(SVector{2}(size(img1)), flow, 1.0, 3.0, difference)

    println("Horizontal Motion")
    println("Error Points Percentage = ", (error_pts/(size(img1)[1]*size(img1)[2]))*100)
    @test ((error_pts/(size(img1)[1]*size(img1)[2]))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
end
