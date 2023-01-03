# importing pycairo
import cairo
# creating a SVG surface
# here geek95 is file name & 700, 700 is dimension
with cairo.SVGSurface("geek95.svg", 700, 700) as surface:
	# creating a cairo context object for SVG surface
	# using Context method
	context = cairo.Context(surface)
	# move the context to x,y position
	context.move_to(50, 200)
	# Drawing Curve
	context.curve_to(150, 75, 225, 50, 350, 150)
	# setting color of the context
	context.set_source_rgb(1, 0, 0)
	# setting width of the context
	context.set_line_width(4)
	# stroke out the color and width property
	context.stroke()
# printing message when file is saved
print("File Saved")
# ---------------------------------------------------------------------------------------
import cv2
import numpy as np
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)
img = cv2.imread('image.jpg')
gammaImg = gammaCorrection(img, 2.2)
cv2.imshow('Original image', img)
cv2.imshow('Gamma corrected image', gammaImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, restoration
rng = np.random.default_rng()
astro = color.rgb2gray(data.astronaut())
from scipy.signal import convolve2d as conv2
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)
deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)
plt.gray()
ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')
ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')
fig.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------
import cv2
import numpy as np
# Read source image.
im_src = cv2.imread('image1.png')
# Four corners of the book in source image
pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
# Read destination image.
im_dst = cv2.imread('image.jpg')
# Four corners of the book in destination image.
pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)
# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
# Display images
cv2.imshow("Source Image", im_src)
cv2.imshow("Destination Image", im_dst)
cv2.imshow("Warped Source Image", im_out)
cv2.waitKey(0)
#---------------------------------------------------------------------------
import cv2
import numpy as np
image = cv2.imread('image.jpg')
#Apply identity kernel
kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
# filter2D() function can be used to apply kernel to an image.
# Where ddepth is the desired depth of final image. ddepth is -1 if...
# ... depth is same as original or source image.
identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
# We should get the same image
cv2.imshow('Original', image)
cv2.imshow('Identity', identity)
cv2.waitKey()
cv2.imwrite('identity.jpg', identity)
cv2.destroyAllWindows()
# ------------------------------------------------------------
import cv2
import numpy as np
image = cv2.imread('image.jpg')
#Apply blurring kernel
kernel2 = np.ones((5, 5), np.float32) / 25
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
cv2.imshow('Original', image)
cv2.imshow('Kernel Blur', img)
cv2.waitKey()
cv2.imwrite('blur_kernel.jpg', img)
cv2.destroyAllWindows()



