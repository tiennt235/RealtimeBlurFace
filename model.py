import numpy as np
import time
import cv2 as cv
import os
import pathlib

def load_face_models():

	# Link: https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD

	txt_file = "Resnet_SSD_deploy.prototxt"
	weight_file = "Res10_300x300_SSD_iter_140000.caffemodel"

	model = cv.dnn.readNet(txt_file, weight_file)
	return model

def pixelate_image(image, grid_size):
	# Chia anh thanh block x block o vuong
	(h, w) = image.shape[:2]
	xGridLines = np.linspace(0, w, grid_size + 1, dtype="int")
	yGridLines = np.linspace(0, h, grid_size + 1, dtype="int")

	# Lap qua tung o vuong
	for i in range(1, len(xGridLines)):
		for j in range(1, len(yGridLines)):

			# Lay toa do cua o vuong hien tai
			cell_startX = xGridLines[j - 1]
			cell_startY = yGridLines[i - 1]
			cell_endX = xGridLines[j]
			cell_endY = yGridLines[i]

			# Trich vung anh theo toa do ben tren
			cell = image[cell_startY:cell_endY, cell_startX:cell_endX]

			# Tinh trung binh cong vung anh va ve vao o vuong hien tai
			(B, G, R) = [int(x) for x in cv.mean(cell)[:3]]
			cv.rectangle(image, (cell_startX, cell_startY), (cell_endX, cell_endY),
				(B, G, R), -1)

	return image

def blur_face(image):
    threshold = 0.5  # objects' confidence threshold

    model = load_face_models()

    if image is None:
        print('Image not loaded.')
        return None
        
    (h, w) = image.shape[:2]

    # Phat hien khuon mat
    blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    # Lap qua ket qua dau ra
    for i in range(0, detections.shape[2]):

        # Lay confidence
        confidence = detections[0, 0, i, 2]
        # print(detections[0, 0, i])
        # Neu confiden > 0.5 moi xu ly
        if (detections[0, 0, i, 1] == 1) and (confidence > threshold):
            # Lay toa do that
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Lay phan khuon mat
            face = image[startY:endY, startX:endX]

            # Pixelate
            face = pixelate_image(face, grid_size=int(w * 0.01))
            # Ve de phan pixelate len
            image[startY:endY, startX:endX] = face

    return image