import numpy as np
from PIL import Image
from torchvision import transforms
import os
from piqa import FSIM
import cv2

def _compute_fsim(a, b, max_val: float = 255.0, **kwargs) -> float:
	fsim = FSIM().eval()
	return fsim(a / max_val, b / max_val).item()


def get_crop(pil_image, poly):
	img= cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
	cnt = np.array([
			[[int(poly[0]), int(poly[1])]],
			[[int(poly[2]), int(poly[3])]],
			[[int(poly[4]), int(poly[5])]],
			[[int(poly[6]), int(poly[7])]]
		])
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	img_crop, img_rot = crop_rect(img, rect)
	img_crop= cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
	return Image.fromarray(img_crop)


def crop_rect(img, rect):
	center, size, angle = rect[0], rect[1], rect[2]
	center, size = tuple(map(int, center)), tuple(map(int, size))
	height, width = img.shape[0], img.shape[1]
	M = cv2.getRotationMatrix2D(center, angle, 1)
	img_rot = cv2.warpAffine(img, M, (width, height))
	img_crop = cv2.getRectSubPix(img_rot, size, center)
	return img_crop, img_rot

def process(poly, img):
	if poly == []:
		return [0, 0, 0, 0]
	image_width, image_height = img.size
	crop_size = 300
	center_x = poly[0] + (poly[6] - poly[0]) / 2
	center_y = poly[3] + (poly[1] - poly[3]) / 2

	left = max(0, center_x - crop_size // 2)
	top = max(0, center_y - crop_size // 2)
	right = left + crop_size
	bottom = top + crop_size
	if right > image_width:
		left = max(0, image_width - crop_size)
		right = image_width

	if bottom > image_height:
		top = max(0, image_height - crop_size)
		bottom = image_height

	return [left, top, right, bottom]



def get_infer(device='cpu'):
	from mmocr.apis import MMOCRInferencer
	return MMOCRInferencer(det='dbnetpp', device = device)


def text_metric(orig_path, ai_path_list,  classic_path_list, device='cuda', infer='no_infer'):
	if infer == 'no_infer':
		try:
			from mmocr.apis import MMOCRInferencer
			infer = MMOCRInferencer(det='dbnetpp', device=device)
		except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError, AttributeError) as e:
			error_msg = str(e)
			if '__version__' in error_msg or 'mmcv' in error_msg.lower():
				raise RuntimeError(f"MMOCR requires properly installed mmcv (>=2.0.0rc4,<2.1.0). "
								   f"Please reinstall mmcv. Error: {error_msg}")
			raise RuntimeError(f"Failed to initialize MMOCR inferencer. Error: {error_msg}")

	block_size = 300
	conf = 0.7
	def get_name(path):
		return os.path.splitext(os.path.basename(path))[0]

	img_name = get_name(orig_path)
	res_dict = {img_name: []}

	im_orig = Image.open(orig_path).convert("RGB")
	w_orig, h_orig = im_orig.size
	result = infer(orig_path, return_vis=False)
	polys = result['predictions'][0]['det_polygons']
	scores = result['predictions'][0]['det_scores']

	for i in range(len(classic_path_list)):
		classic_image = Image.open(classic_path_list[i]).convert("RGB")
		ai_image = Image.open(ai_path_list[i]).convert("RGB")
		max_d_fsim = 0
		poly_max = []

		for j in range(len(polys)):
			polygon = polys[j]
			if scores[j] > conf:
				crop_orig = get_crop(im_orig, polygon)
				crop_neur = get_crop(ai_image, polygon)
				crop_vtm = get_crop(classic_image, polygon)

				w, h = crop_orig.size
				X = transforms.ToTensor()(crop_orig).unsqueeze(0) * 255
				Y_vtm = transforms.ToTensor()(crop_vtm).unsqueeze(0) * 255
				Y_neur = transforms.ToTensor()(crop_neur).unsqueeze(0) * 255

				metric_vtm = _compute_fsim(X, Y_vtm)
				metric_neur = _compute_fsim(X, Y_neur)
				d_fsim = metric_vtm - metric_neur

				if d_fsim > max_d_fsim:
					max_d_fsim = d_fsim
					poly_max = polygon

		bbox = process(poly_max, im_orig) if poly_max else [0, 0, 0, 0]
		res_dict[img_name].append({
			'metric_value': max_d_fsim,
			'bbox': bbox,
			'ai_path': ai_path_list[i],
			'classic_path': classic_path_list[i],
			'gt_path': orig_path,
		})

	return res_dict

