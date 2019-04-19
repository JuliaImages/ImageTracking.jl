using Images
using ImageTracking:sample_roi

@testset "ROI sampling" begin
    @info "Running ROI sampling tests"
    box = BoxROI(ones(300, 300), [125, 125, 175, 175])
    sampler_overlap = 0.5
    search_factor = 2.0
    int_img = integral_image(box.img)
    currentsample = CurrentSampler(sampler_overlap, search_factor)
    # positive sampling
    currentsample.mode = :positive
    @time samples = sample_roi(currentsample, BoxROI(int_img, box.bound))

    @test size(samples) == (1, 4)
    @test samples[1][1] == int_img[125:175, 125:175]

    # negative sampling
    currentsample.mode = :negative
    @time samples = sample_roi(currentsample, BoxROI(int_img, box.bound))

    @test size(samples) == (1, 4)
    @test samples[1][1] == int_img[100:150, 100:150]
    @test samples[2][1] == int_img[100:150, 151:201]
    @test samples[3][1] == int_img[151:201, 100:150]
    @test samples[4][1] == int_img[151:201, 151:201]

end;
