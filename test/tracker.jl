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
end
