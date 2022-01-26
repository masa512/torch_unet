import numpy as np
import cv2
from tifffile import imread, imsave
import matplotlib.pyplot as plt
import os
from skimage.filters import try_all_threshold


_dir = r"F:\Bala Bala Masa\torch_unet\Focus20220123-105048\val_outputs\gt_img"
_iname = 'image44.tif'
I = imread(os.path.join(_dir,_iname))[0,:,:]
_th = 0.5005
min_mask = 0

# Try all shit
try_all_threshold(I)
plt.show()

'''
G = cv2.GaussianBlur(I, (9,9), 0)
M = cv2.threshold(src = G, thresh = _th, maxval = 1-min_mask, type = cv2.THRESH_BINARY)[1].astype(np.uint8)+min_mask


# Make the contours
c = cv2.findContours(M,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]


t = np.zeros_like(M)

for i, cnt in enumerate(c):
	if cv2.contourArea(cnt) > 1500:
		cv2.drawContours(t, c, i, (255, 255, 255),thickness=cv2.FILLED)


plt.subplot(1,2,1)
plt.imshow(t)

plt.subplot(1,2,2)
plt.imshow(I)
plt.show()
'''



# Main shit
'''
def rescale_mask(mask = None, _min: int = 0, _max: int = 1):
	M,N = mask.shape
	return np.array([[_min if mask[i,j] == 0 else _max for j in range(N)] for i in range(M)])


if __name__ == '__main__':

	_dir = r"./../Hela_data/In_Focus"
	_odir = os.path.join(_dir,'mask')
	if not os.path.exists(_odir):
		os.mkdir(_odir)

	_th = 0.02
	min_mask = 0
	files = os.listdir(_dir)
	for f in files:
		if f.endswith('.tif') and not f.startswith('.'):
			I = imread(os.path.join(_dir,f))

			
			G = cv2.GaussianBlur(I, (9,9), 0)
			M = cv2.threshold(src = G, thresh = _th, maxval = 1, type = cv2.THRESH_BINARY)[1].astype(np.uint8)
			M_m = cv2.threshold(src = G, thresh = _th, maxval = 1-min_mask, type = cv2.THRESH_BINARY)[1].astype(np.float32)+min_mask
			# Make the contours
			c = cv2.findContours(M,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]


			t = np.zeros_like(M)

			for i, cnt in enumerate(c):
				if cv2.contourArea(cnt) > 500:
					cv2.drawContours(t, c, i, (1, 1, 1),thickness=cv2.FILLED)
			imsave(os.path.join(_odir,f),t.astype(np.float32)*(1-min_mask)+min_mask)
'''






