import numpy as np
import skimage.measure
import time
import matplotlib.pyplot as plt
def point_inside(box, point):

	return point[0] >= box[0] and point[0] <= (box[0] + box[2]) and point[1] >= box[1] and point[1] <= (box[1] + box[3])


# Given two boxes, determines overlap percentage

def overlap_percentage(box1, box2):
	if contains(box1, box2) or contains(box2, box1):
		return 1.0
	tl, tr, bl, br = False

	if point_inside(box1, [box2[0], box2[1]]):
		tl = True
	if point_inside(box1, [box2[0] + box2[2], box2[1]]):
		bl = True
	if point_inside(box1, [box2[0], box2[1] + box2[3]]):
		tr = True
	if point_inside(box1, [box2[0] + box2[2], box2[1] + box2[3]]):
		br = True

	# Eight possible cases
	diffx = 0
	diffy = 0

	# No nodes
	if not (tl or bl or tr or br):
		return 0.0

	# Left nodes
	elif tl and bl:
		diffx = box2[2]
		diffy = box1[1] + box1[3] - box2[1]

	elif tl and tr:
		diffy = box2[3]
		diffx = box1[0] + box1[2] - box2[0]

	elif bl and br:
		diffy = box2[3]
		diffx = box2[0] - box1[0]

	elif br and tr:
		diffx = box2[2]
		diffy = box2[1] - box1[1]

	elif tl:
		diffx = box1[0] + box1[2] - box2[0]
		diffy = box1[1] + box1[3] - box2[1]

	elif tr:
		diffx = box2[0] + box2[2] - box1[0]
		diffy = box1[1] + box1[3] - box2[1]

	elif bl:
		diffx = box2[0] + box2[2] - box1[0]
		diffy = box1[1] + box1[3] - box2[1]

	elif br:
		diffx = box2[0] + box2[2] - box1[0]
		diffy = box2[1] + box2[3] - box1[0]



	overlap = diffx * diffy


	return overlap / (box1[2]*box1[3] + box2[2]*box1[3] - overlap)





# Check if box1 is located inside of box2
def contains(box1, box2):

	# Need to check all 4 coordinates
	if box1[0] >= box2[0] and (box1[0] + box1[2]) <= (box2[0] + box2[2]):
		if box1[1] >= box2[1] and (box1[1] + box1[3]) <= (box2[1] + box2[3]):
			return True

	return False




def predict_bb(length, width, image, threshold, overlap_threshold):

	# Pool the probabilities into less than 32 * 32
	l = len(image)
	w = len(image[0])
	pool = image[:]
	while (l > length and w > width):
		
		pool = skimage.measure.block_reduce(pool, (2,2), np.mean)

		l = l / 2
		w = w / 2


	l = int(l)
	w = int(w)
	# Determine bounding boxes probabilities
	bb_shape = [[5,3],[5,4], [5,5], [7,3], [7,4], [10, 4], [10, 6], [10,8], [16, 8]]

	bounding_boxes = []
	for i in bb_shape:

		for x in range(l - i[0] + 1):
			for y in range(w - i[1] + 1):

				probability = np.mean(image[x: x + i[0], y: y + i[1]])

				if probability > threshold:

					# Top left coordinate (x,y), length, width, probability
					new_box = [x,y,i[0], i[1], probability]

					# Remove boxes that are located inside of the new_box or don't add box
					# if the current box is inside another box

					new_bounding_boxes = []
					new_box_inside = False
					for box in bounding_boxes:

						if not contains(box, new_box):
							new_bounding_boxes.append(box)

						elif contains(new_box, box):
							new_box_inside = True

					if not new_box_inside:
						new_bounding_boxes.append(new_box)

					bounding_boxes = new_bounding_boxes
	# Merge boxes
	# loop = True

	# while (loop):
	# 	loop = False

	# 	for i in range(len(bounding_boxes)):
	# 		for j in range(len(bounding_boxes)):
	# 			if i != j:
	# 				overlap_percent = max(overlap_percentage(bounding_boxes[i], bounding_boxes[j]), overlap_percentage(bounding_boxes[j], bounding_boxes[i]))
	# 				if overlap_percent > overlap_threshold:
	# 					loop = True

	# 					bounding_boxes = 

	# 					break
	# 		if loop == True:
	# 			break

	# if boxes overlap for a certain threshold, merge them
		# pick best two boxes at the end if two boxes still exceed threshold

	bounding_boxes.sort(key = lambda x: x[4], reverse = True)
	num_boxes = min(3,len(bounding_boxes))

	if num_boxes == 0:
		return []
	else:
		predicted_boxes = bounding_boxes[:num_boxes]

		for p in predicted_boxes:
			lmult = len(image)/l
			wmult = len(image[0])/w

			p[0] = p[0] * lmult
			p[1] = p[1] * wmult
			p[2] = p[2] * lmult
			p[3] = p[3] * wmult

		return predicted_boxes
		# I know the current coordinates


	# Determine coordinates of the boxes in the actual image if they exist

	# Return them in the appropriate format



if __name__ == "__main__":

	x = time.time()
	zer = np.random.normal(0,1,[1024,1024])
	print(zer.shape)

	output = predict_bb(32, 32, zer, 0.7, 0.5)
	y = time.time()
	plt.imshow(zer * 255)
	print(y-x)
	print("-____----")
	print(output)
	plt.show()
