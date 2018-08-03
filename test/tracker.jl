using Images, TestImages, StaticArrays

function find_overlap(tracker_bounding_box::MVector{4, Int}, ground_truth_bounding_box::MVector{4, Int})
	top = max(tracker_bounding_box[1], ground_truth_bounding_box[1])
	bottom = min(tracker_bounding_box[3], ground_truth_bounding_box[3])
	left = max(tracker_bounding_box[2], ground_truth_bounding_box[2])
	right = min(tracker_bounding_box[4], ground_truth_bounding_box[4])

	overlap_area = (bottom - top + 1)*(right - left + 1)
	ground_truth_area = (ground_truth_bounding_box[3] - ground_truth_bounding_box[1] + 1)*(ground_truth_bounding_box[4] - ground_truth_bounding_box[2] + 1)

	return (overlap_area/ground_truth_area)*100
end

@testset "MedianFlow" begin

	#Simple tracking sequence
	tr = TrackerMedianFlow()
	a = MVector{4}(248,168,276,196)
	img = Gray{Float64}.(testimage("lake"))
	img[250:274, 170:194] .= 0
	init_tracker(tr, img, a)

	img = Gray{Float64}.(testimage("lake"))
	img[253:277, 175:199] .= 0
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(251, 173, 279, 201))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(testimage("lake"))
	img[254:278, 177:201] .= 0
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(253, 176, 279, 202))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(testimage("lake"))
	img[257:281, 181:205] .= 0
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(256, 180, 282, 206))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(testimage("lake"))
	img[262:286, 178:202] .= 0
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(261, 177, 287, 203))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(testimage("lake"))
	img[265:289, 176:200] .= 0
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(264, 175, 290, 201))
	@show overlap_percentage
	@test overlap_percentage > 50

	#Real data tracking sequence
	tr = TrackerMedianFlow()
	a = MVector{4}(81,150,120,173)
	img = Gray{Float64}.(load("test_data/tracking/images/img00000.png"))
	init_tracker(tr, img, a)

	img = Gray{Float64}.(load("test_data/tracking/images/img00001.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(83, 150, 122, 173))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00002.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(84, 151, 123, 174))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00003.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(85, 151, 124, 174))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00004.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(87, 151, 126, 174))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00005.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(89, 152, 128, 175))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00006.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(90, 155, 129, 178))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00007.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(90, 156, 129, 179))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00008.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(91, 158, 130, 181))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00009.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(90, 160, 129, 183))
	@show overlap_percentage
	@test overlap_percentage > 50

	img = Gray{Float64}.(load("test_data/tracking/images/img00010.png"))
	bounding_box = update_tracker(tr, img)
	overlap_percentage = find_overlap(bounding_box, MVector{4, Int}(87, 161, 126, 184))
	@show overlap_percentage
	@test overlap_percentage > 50
end
