from PIL import Image
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.image as mi

#pip install scikit-learn

from sklearn.cluster import KMeans
# Open Image
Image.open('tifa.jpg')


read_image = mi.imread('tifa.jpg')
width, height, depth = tuple(read_image.shape)
read_pixels = np.reshape(read_image, (width * height, depth))

n_colors = 10
color_extracting_model = KMeans(n_clusters=n_colors, random_state=42).fit(read_pixels)

color_palette = np.uint8(color_extracting_model.cluster_centers_)
pp.imshow([color_palette])
pp.show()




