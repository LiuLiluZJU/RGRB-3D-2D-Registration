import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

x_ray_ap = sitk.ReadImage("/home/leko/Desktop/x_ray1.DCM")
x_ray_lat = sitk.ReadImage("/home/leko/Desktop/x_ray2.DCM")
x_ray_ap_array = sitk.GetArrayFromImage(x_ray_ap)
x_ray_ap_array = ndimage.filters.gaussian_filter(x_ray_ap_array.copy(), 0.5 / 0.143)  # Gaussian filter
x_ray_lat_array = sitk.GetArrayFromImage(x_ray_lat)
x_ray_lat_array = ndimage.filters.gaussian_filter(x_ray_lat_array.copy(), 0.5 / 0.143)  # Gaussian filter
x_ray_ap_array = np.squeeze(x_ray_ap_array).astype(np.float64)
x_ray_lat_array = np.squeeze(x_ray_lat_array).astype(np.float64)
x_ray_lat_array = x_ray_lat_array[:, ::-1]
print(x_ray_ap.GetSize())
print(x_ray_lat.GetSize())
size_x_ap, size_y_ap, size_z_ap = x_ray_ap.GetSize()
size_x_lat, size_y_lat, size_z_lat = x_ray_lat.GetSize()
x_ray_ap_array_padded = np.pad(x_ray_ap_array, pad_width=1, mode='edge')
x_ray_lat_array_padded = np.pad(x_ray_lat_array, pad_width=1, mode='edge')

x_ray_ap_grad_array = np.zeros((size_y_ap, size_x_ap, 2))
x_ray_lat_grad_array = np.zeros((size_y_lat, size_x_lat, 2))
print(x_ray_ap_array_padded.dtype)
print(x_ray_ap_array_padded.dtype)
for i in range(size_x_ap):
    for j in range(size_y_ap):
# for i in range(1100, 1600):
#     for j in range(2100, 2988):
# for i in range(1150, 1550):
#     for j in range(1400, 2000):
        x_ray_ap_grad_array[j][i][0] = (x_ray_ap_array_padded[j][i + 2] + 2 * x_ray_ap_array_padded[j + 1][i + 2] + x_ray_ap_array_padded[j + 2][i + 2]) - \
                                       (x_ray_ap_array_padded[j][i] + 2 * x_ray_ap_array_padded[j + 1][i] + x_ray_ap_array_padded[j + 2][i])
        x_ray_ap_grad_array[j][i][1] = (x_ray_ap_array_padded[j + 2][i] + 2 * x_ray_ap_array_padded[j + 2][i + 1] + x_ray_ap_array_padded[j + 2][i + 2]) - \
                                       (x_ray_ap_array_padded[j][i] + 2 * x_ray_ap_array_padded[j][i + 1] + x_ray_ap_array_padded[j][i + 2])
for i in range(size_x_lat):
    for j in range(size_y_lat):
# for i in range((size_x_lat - 1300), (size_x_lat - 700)):
#     for j in range(2100, 2984):
# for i in range((size_x_lat - 1000), (size_x_lat - 600)):
#     for j in range(1400, 2000):
        x_ray_lat_grad_array[j][i][0] = (x_ray_lat_array_padded[j][i + 2] + 2 * x_ray_lat_array_padded[j + 1][i + 2] + x_ray_lat_array_padded[j + 2][i + 2]) - \
                                       (x_ray_lat_array_padded[j][i] + 2 * x_ray_lat_array_padded[j + 1][i] + x_ray_lat_array_padded[j + 2][i])
        x_ray_lat_grad_array[j][i][1] = (x_ray_lat_array_padded[j + 2][i] + 2 * x_ray_lat_array_padded[j + 2][i + 1] + x_ray_lat_array_padded[j + 2][i + 2]) - \
                                       (x_ray_lat_array_padded[j][i] + 2 * x_ray_lat_array_padded[j][i + 1] + x_ray_lat_array_padded[j][i + 2])
x_ray_ap_grad_show = np.squeeze(np.linalg.norm(x_ray_ap_grad_array, axis=2))
plt.imshow(x_ray_ap_grad_show, cmap='gray')
plt.show()
np.save("x_ray_ap_grad.npy", x_ray_ap_grad_array)
np.save("x_ray_lat_grad.npy", x_ray_lat_grad_array)