import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect


if __name__ == '__main__':

	try:
	
		input_dir  = '/Users/adibasubah/alpr-unconstrained/test_ball'
		output_dir = '/Users/adibasubah/alpr-unconstrained/dekha_jak'

		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		# vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		# vehicle_meta = dn.load_meta(vehicle_dataset)
		vehicle_net  = dn.load_net(vehicle_netcfg.encode("utf-8"), 
                           vehicle_weights.encode("utf-8"), 
                           0)

		vehicle_meta = dn.load_meta(vehicle_dataset.encode("utf-8"))

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		print('imgs_paths',imgs_paths)

		if not isdir(output_dir):
			makedirs(output_dir)

		# print 'Searching for vehicles using YOLO...'

		for i,img_path in enumerate(imgs_paths):

			bname = basename(splitext(img_path)[0])

			# R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)
			R, _ = detect(vehicle_net, vehicle_meta, img_path.encode("utf-8"), thresh=vehicle_threshold)
			print("Detection Results:", R) 

			# R = [r for r in R if r[0] in ['car','bus']]
			R = [r for r in R if r[0].decode('utf-8') in ['car', 'bus']]

			print('len_R: ' , len(R))

			if len(R):

				Iorig = cv2.imread(img_path)
				WH = np.array(Iorig.shape[1::-1],dtype=float)
				print('WH ' , WH)
				Lcars = []

				for i, r in enumerate(R):

					cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
					x1, y1 = int((cx - w / 2) * WH[0]), int((cy - h / 2) * WH[1])
					x2, y2 = int((cx + w / 2) * WH[0]), int((cy + h / 2) * WH[1])

					# Draw bounding box
					cv2.rectangle(Iorig, (x1, y1), (x2, y2), (0, 255, 0), 2)

					# Add label text
					text = f"{r[0].decode()} {r[1]:.2f}"  # Decode class name and add confidence score
					cv2.putText(Iorig, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					# Append label
					label = Label(0, np.array([x1, y1]), np.array([x2, y2]))
					Lcars.append(label)  # Fix: Append detected objects to Lcars


				# Save the image with bounding boxes
				output_path = f"{output_dir}/{bname}_detected.png"
				cv2.imwrite(output_path, Iorig)
				print(f"Saved detected image: {output_path}")

				lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	