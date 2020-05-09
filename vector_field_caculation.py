import SimpleITK as sitk
import numpy as np

x_ray_ap = sitk.ReadImage("/home/leko/Desktop/x_ray1.DCM")
x_ray_lat = sitk.ReadImage("/home/leko/Desktop/x_ray2.DCM")
spacing_x_ap, spacing_y_ap = x_ray_ap.GetSpacing()
spacing_x_lat, spacing_y_lat = x_ray_lat.GetSpacing()
size_x_ap, size_y_ap, size_z_ap = x_ray_ap.GetSize()
size_x_lat, size_y_lat, size_z_lat = x_ray_lat.GetSize()

DRR_ap = sitk.Image()
