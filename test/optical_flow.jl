using Images, TestImages

# Testing constants
test_image = "mandrill"
number_test_pts = 200
precision = 0.1
max_error_points_percentage = 10
max_allowed_error = 1.5
max_lost_points_percentage = 5

function test_lk(number_test_pts::Int64, flow::Array{Coordinate{Float64}, 1}, x_flow::Float64, y_flow::Float64, status::BitArray{1}, err::Array{Float64, 1}, precision::Float64)
    #Testing parameters
    error_pts = 0
    max_err = 0
    total_err = 0
    lost_points = 0

    for i = 1:number_test_pts
        if status[i]
            if abs(flow[i].x - x_flow) > precision || abs(flow[i].y - y_flow) > precision
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
    img1 = testimage(test_image)
    img2 = similar(img1)
    for i = 1:size(img1)[1]
        for j = 4:size(img1)[2]
            img2[i,j] = img1[i,j-3]
        end
    end
    test_img1 = Gray.(img1)
    test_img2 = Gray.(img2)
    corners = imcorner(test_img1, method=shi_tomasi)
    a = [Coordinate(1,1)]
    for i = 1:size(img1)[1]
        for j = 1:size(img1)[2]
            if corners[i,j]
                append!(a, [Coordinate(j,i)])
            end
        end
    end
    a = deleteat!(a, 1)

    pts = rand(a, (number_test_pts,))
    flow, status, err = optical_flow(test_img1, test_img2, LK(pts, [Coordinate(0.0,0.0)], 5, 4, false, 20))

    error_pts, max_err, total_err, lost_points = test_lk(number_test_pts, flow, 3.0, 0.0, status, err, precision)

    println("Horizontal Motion")
    println("Error Points Percentage = ", (error_pts/(size(a)[1] - lost_points))*100)
    @test ((error_pts/(size(a)[1] - lost_points))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
    @test max_err < max_allowed_error
    println("Lost Points Percentage = ", (lost_points/size(a)[1])*100)
    @test ((lost_points/size(a)[1])*100) < max_lost_points_percentage


    # Basic translations (Vertical)
    img2 = similar(img1)
    for i = 4:size(img1)[1]
        for j = 1:size(img1)[2]
            img2[i,j] = img1[i-3,j]
        end
    end
    test_img2 = Gray.(img2)

    pts = rand(a, (number_test_pts,))
    flow, status, err = optical_flow(test_img1, test_img2, LK(pts, [Coordinate(0.0,0.0)], 5, 4, false, 20))

    error_pts, max_err, total_err, lost_points = test_lk(number_test_pts, flow, 0.0, 3.0, status, err, precision)

    println("Vertical Motion")
    println("Error Points Percentage = ", (error_pts/(size(a)[1] - lost_points))*100)
    @test ((error_pts/(size(a)[1] - lost_points))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
    @test max_err < max_allowed_error
    println("Lost Points Percentage = ", (lost_points/size(a)[1])*100)
    @test ((lost_points/size(a)[1])*100) < max_lost_points_percentage


    # Basic translations (Both)
    img2 = similar(img1)
    for i = 4:size(img1)[1]
        for j = 2:size(img1)[2]
            img2[i,j] = img1[i-3,j-1]
        end
    end
    test_img2 = Gray.(img2)

    pts = rand(a, (number_test_pts,))
    flow, status, err = optical_flow(test_img1, test_img2, LK(pts, [Coordinate(0.0,0.0)], 5, 4, false, 20))

    error_pts, max_err, total_err, lost_points = test_lk(number_test_pts, flow, 1.0, 3.0, status, err, precision)

    println("Combined Motion")
    println("Error Points Percentage = ", (error_pts/(size(a)[1] - lost_points))*100)
    @test ((error_pts/(size(a)[1] - lost_points))*100) < max_error_points_percentage
    println("Maximum Error = ", max_err)
    @test max_err < max_allowed_error
    println("Lost Points Percentage = ", (lost_points/size(a)[1])*100)
    @test ((lost_points/size(a)[1])*100) < max_lost_points_percentage
end
