import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.measure import ransac
from skimage.transform import AffineTransform
import random


def augment_matrix_coord(array):

    n = len(array)
    return np.concatenate((array, np.ones((n,1))), axis = 1).T


def get_rotation_mat_single_axis( axis, angle ):

    """It computes the 3X3 rotation matrix relative to a single rotation of angle(rad) 
    about the axis(string 'x', 'y', 'z') for a righr handed CS"""

    if axis == 'x' : return np.array(([1,0,0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]))

    if axis == 'y' : return np.array(([np.cos(angle),0,np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]))

    if axis == 'z' : return np.array(([np.cos(angle),-np.sin(angle),0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]))


def get_rigid_motion_mat_from_euler( alpha, axis_1, beta, axis_2, gamma, axis_3, t_x, t_y, t_z ):
    
    """It computes the 4X4 rigid motion matrix given a sequence of 3 Euler angles about the 3 axes 1,2,3 
    and the translation vector t_x, t_y, t_z"""

    rot1 = get_rotation_mat_single_axis( axis_1, alpha )
    rot2 = get_rotation_mat_single_axis( axis_2, beta )
    rot3 = get_rotation_mat_single_axis( axis_3, gamma )

    rot_mat = np.dot(rot1, np.dot(rot2,rot3))

    t = np.array(([t_x], [t_y], [t_z]))

    output = np.concatenate((rot_mat, t), axis = 1)

    return np.concatenate((output, np.array([[0.,0.,0.,1.]])), axis = 0)


CT_in = sitk.ReadImage("/home/leko/Desktop/CT1.mha")
CT_in_array = sitk.GetArrayFromImage(CT_in)
CT_in_array = ndimage.filters.gaussian_filter(CT_in_array.copy(), (0.3 / 2, 0.3 / 0.68, 0.3 / 0.68))  # Gaussian filter
PhysicalPointImagefilter = sitk.PhysicalPointImageSource()
PhysicalPointImagefilter.SetReferenceImage(CT_in)
CT_in_cartesian = PhysicalPointImagefilter.Execute()
CT_in_cartesian_array = sitk.GetArrayFromImage(CT_in_cartesian)
print(CT_in_cartesian_array[0][0][0])
print(CT_in_cartesian_array[158][511][511])
origin = CT_in.GetOrigin()
print(origin)
size_x, size_y, size_z = CT_in.GetSize()
spacing_x, spacing_y, spacing_z = CT_in.GetSpacing()
focal_length = 2000
center = np.zeros(3)
center[0] = origin[0] + ((size_x - 1) * spacing_x) / 2
center[1] = origin[1] + ((size_y - 1) * spacing_y) / 2
center[2] = origin[2] + ((size_z - 1) * spacing_z) / 2
source = np.zeros(3)
source[0] = center[0]
source[1] = center[1]
source[2] = center[2] - focal_length / 2

# Calculate CT's gradient
# CT_grad_array = np.zeros((size_z, size_y, size_x, 3))
# CT_in_array_padded = np.pad(CT_in_array, pad_width=1, mode='edge')
# print(CT_in_array_padded.shape)
# for i in range(size_x):
#     for j in range(size_y):
#         for k in range(size_z):
#             # # Transform Cartesian coordinate to spherical coordinate
#             # # r
#             # R = np.linalg.norm(CT_in_cartesian_array[k][j][i] - source)
#             # # phi
#             # phi = np.arctan(CT_in_cartesian_array[k][j][i][1] / CT_in_cartesian_array[k][j][i][0])
#             # # theta
#             # theta = np.arcsin(CT_in_cartesian_array[k][j][i][0] / (np.cos(phi) * R))
#             CT_grad_array[k][j][i][0] = CT_in_array_padded[k + 1][j + 1][i + 2] - CT_in_array_padded[k + 1][j + 1][i]
#             CT_grad_array[k][j][i][1] = CT_in_array_padded[k + 1][j + 2][i + 1] - CT_in_array_padded[k + 1][j][i + 1]
#             CT_grad_array[k][j][i][2] = CT_in_array_padded[k + 2][j + 1][i + 1] - CT_in_array_padded[k][j + 1][i + 1]
# np.save("3d_grad.npy", CT_grad_array)
CT_grad_array = np.load("3d_grad.npy")
CT_grad_norm = np.linalg.norm(CT_grad_array, axis=3)
# CT_grad_norm = CT_grad_norm[::-1, ::-1, :]
CT_grad_mean = np.mean(CT_grad_norm, 2)
print(CT_grad_mean.shape)
plt.imshow(CT_grad_mean, cmap='gray')
plt.show()

# Calculate vector field
x_ray_ap = sitk.ReadImage("/home/leko/Desktop/x_ray1.DCM")
x_ray_lat = sitk.ReadImage("/home/leko/Desktop/x_ray2.DCM")
spacing_x_ap, spacing_y_ap, spacing_z_ap = x_ray_ap.GetSpacing()
spacing_x_lat, spacing_y_lat, spacing_z_lat = x_ray_lat.GetSpacing()
size_x_ap, size_y_ap, size_z_ap = x_ray_ap.GetSize()
size_x_lat, size_y_lat, size_z_lat = x_ray_lat.GetSize()

# AP physical coordinate
DRR_ap = sitk.Image([size_x_ap, size_y_ap, 1], sitk.sitkFloat64)
DRRorigin_ap = (center[0] - spacing_x_ap * (size_x_ap - 1) / 2,
                center[1] - spacing_y_ap * (size_y_ap - 1) / 2,
                center[2] + focal_length / 2)
DRRspacing_ap = (spacing_x_ap, spacing_y_ap, 1)
CT_in_direction = CT_in.GetDirection()
print(CT_in_direction)
DRR_ap.SetOrigin(DRRorigin_ap)
DRR_ap.SetSpacing(DRRspacing_ap)
DRR_ap.SetDirection(CT_in_direction)
PhysicalPointImagefilter_ap = sitk.PhysicalPointImageSource()
PhysicalPointImagefilter_ap.SetReferenceImage(DRR_ap)
DRR_ap_cartesian = PhysicalPointImagefilter_ap.Execute()
DRR_ap_cartesian_array = sitk.GetArrayFromImage(DRR_ap_cartesian)
print(DRR_ap_cartesian_array.shape)

transform_parameters_ap = [np.deg2rad(90), np.deg2rad(0), np.deg2rad(0), 0, 0, 728]
rotx = transform_parameters_ap[0]
roty = transform_parameters_ap[1]
rotz = transform_parameters_ap[2]
tx = transform_parameters_ap[3]
ty = transform_parameters_ap[4]
tz = transform_parameters_ap[5]
T_ap = get_rigid_motion_mat_from_euler(rotz, 'z', roty, 'y', rotx, 'x', tx, ty, tz)
invT_ap = np.linalg.inv(T_ap)

source_transformed_ap = np.dot(invT_ap, np.array([source[0] - center[0], source[1] - center[1], source[2] - center[2], 1.]).T)[0:3]
source_ap = np.array([source_transformed_ap[0] + center[0], source_transformed_ap[1] + center[1],source_transformed_ap[2] + center[2]])

Tn = np.array([[1., 0., 0., center[0]],
                [0., 1., 0., center[1]],
                [0., 0., 1., center[2]],
                [0., 0., 0., 1.]])
invTn = np.linalg.inv(Tn)

DRR_ap_cartesian_array = np.squeeze(DRR_ap_cartesian_array)
DRR_ap_cartesian_array_reshaped = DRR_ap_cartesian_array.reshape((size_x_ap * size_y_ap, 3), order='C')
DRR_ap_cartesian_array_augmented = np.dot(invTn, augment_matrix_coord(DRR_ap_cartesian_array_reshaped))
DRR_ap_cartesian_array_augmented_transformed = np.dot(invT_ap, DRR_ap_cartesian_array_augmented)
DRR_ap_cartesian_array_transformed = np.transpose(np.dot(Tn, DRR_ap_cartesian_array_augmented_transformed)[0:3])
print(DRR_ap_cartesian_array_transformed.shape)
DRR_ap_cartesian_array_new = np.reshape(DRR_ap_cartesian_array_transformed, (size_y_ap, size_x_ap, 3), order='C')

# LAT physical coordinate
DRR_lat = sitk.Image([size_x_lat, size_y_lat, 1], sitk.sitkFloat64)
DRRorigin_lat = (center[0] - spacing_x_lat * (size_x_lat - 1) / 2,
                center[1] - spacing_y_lat * (size_y_lat - 1) / 2,
                center[2] + focal_length / 2)
print(DRRorigin_lat)
DRRspacing_lat = (spacing_x_lat, spacing_y_lat, 1)
DRR_lat.SetOrigin(DRRorigin_lat)
DRR_lat.SetSpacing(DRRspacing_lat)
DRR_lat.SetDirection(CT_in_direction)
PhysicalPointImagefilter_lat = sitk.PhysicalPointImageSource()
PhysicalPointImagefilter_lat.SetReferenceImage(DRR_lat)
DRR_lat_cartesian = PhysicalPointImagefilter_lat.Execute()
DRR_lat_cartesian_array = sitk.GetArrayFromImage(DRR_lat_cartesian)
print(DRR_lat_cartesian_array.shape)

transform_parameters_lat = [np.deg2rad(90), np.deg2rad(0), np.deg2rad(-90), 0, 0, 699]
rotx = transform_parameters_lat[0]
roty = transform_parameters_lat[1]
rotz = transform_parameters_lat[2]
tx = transform_parameters_lat[3]
ty = transform_parameters_lat[4]
tz = transform_parameters_lat[5]
T_lat = get_rigid_motion_mat_from_euler(rotx, 'x', roty, 'y', rotz, 'z', tx, ty, tz)
invT_lat = np.linalg.inv(T_lat)

source_transformed_lat = np.dot(invT_lat, np.array([source[0] - center[0], source[1] - center[1], source[2] - center[2], 1.]).T)[0:3]
source_lat = np.array([source_transformed_lat[0] + center[0], source_transformed_lat[1] + center[1],source_transformed_lat[2] + center[2]])

DRR_lat_cartesian_array_reshaped = DRR_lat_cartesian_array.reshape((size_x_lat * size_y_lat, 3), order='C')
DRR_lat_cartesian_array_augmented = np.dot(invTn, augment_matrix_coord(DRR_lat_cartesian_array_reshaped))
DRR_lat_cartesian_array_augmented_transformed = np.dot(invT_lat, DRR_lat_cartesian_array_augmented)
DRR_lat_cartesian_array_transformed = np.transpose(np.dot(Tn, DRR_lat_cartesian_array_augmented_transformed)[0:3])
DRR_lat_cartesian_array_new = np.reshape(DRR_lat_cartesian_array_transformed, (size_y_lat, size_x_lat, 3), order='C')

# # Calculate vector field
# x_ray_ap_grad_array = np.load("x_ray_ap_grad.npy")
# x_ray_lat_grad_array = np.load("x_ray_lat_grad.npy")
# x_ray_lat_grad_array_norm = np.linalg.norm(x_ray_lat_grad_array, axis=2)
# print(np.median(x_ray_lat_grad_array_norm))
# x_ray_lat_grad_array_norm[x_ray_lat_grad_array_norm > 2000] = 2000
# plt.imshow(x_ray_lat_grad_array_norm, cmap='gray')
# plt.show()
# DRRorigin_ap_new = DRR_ap_cartesian_array_new[0][0]
# DRRorigin_lat_new = DRR_lat_cartesian_array_new[0][0]
# print(DRRorigin_lat_new)
# vector_field = np.zeros((size_z, size_y, size_x, 3))
# for i in range(size_x):
#     for j in range(size_y):
#         for k in range(size_z):
#             reconstruct_point = CT_in_cartesian_array[k][j][i]
#             project_point_factor_ap = focal_length / (reconstruct_point[1] - source_ap[1])
#             project_point_ap = np.array([
#                 source_ap[0] + (reconstruct_point[0] - source_ap[0]) * project_point_factor_ap,
#                 source_ap[1] + (reconstruct_point[1] - source_ap[1]) * project_point_factor_ap,
#                 source_ap[2] + (reconstruct_point[2] - source_ap[2]) * project_point_factor_ap
#             ])
#             project_point_ap_index = [
#                 int((project_point_ap[0] - DRRorigin_ap_new[0]) / spacing_x_ap),
#                 int((-project_point_ap[2] + DRRorigin_ap_new[2]) / spacing_y_ap)
#             ]
#
#             # if project_point_ap_index[0] >= 1550 or \
#             #     project_point_ap_index[1] >= 2000 or \
#             #     project_point_ap_index[0] < 1150 or \
#             #     project_point_ap_index[1] < 1400:
#             #     continue
#             # if project_point_ap_index[0] >= 1600 or \
#             #     project_point_ap_index[1] >= 2988 or \
#             #     project_point_ap_index[0] < 1100 or \
#             #     project_point_ap_index[1] < 2100:
#             #     continue
#             if project_point_ap_index[0] >= 1600 or \
#                 project_point_ap_index[1] >= 2400 or \
#                 project_point_ap_index[0] < 1100 or \
#                 project_point_ap_index[1] < 2100:
#                 continue
#             # if project_point_ap_index[0] >= size_x_ap or \
#             #     project_point_ap_index[1] >= size_y_ap or \
#             #     project_point_ap_index[0] < 0 or \
#             #     project_point_ap_index[1] < 0:
#             #     continue
#
#             project_point_factor_lat = -focal_length / (reconstruct_point[0] - source_lat[0])
#             project_point_lat = np.array([
#                 source_lat[0] + (reconstruct_point[0] - source_lat[0]) * project_point_factor_lat,
#                 source_lat[1] + (reconstruct_point[1] - source_lat[1]) * project_point_factor_lat,
#                 source_lat[2] + (reconstruct_point[2] - source_lat[2]) * project_point_factor_lat
#             ])
#             project_point_lat_index = [
#                 int((project_point_lat[1] - DRRorigin_lat_new[1]) / spacing_x_lat),
#                 int((-project_point_lat[2] + DRRorigin_lat_new[2]) / spacing_y_lat)
#             ]
#
#             # if project_point_lat_index[0] >= (size_x_lat - 600) or \
#             #     project_point_lat_index[1] >= 2000 or \
#             #     project_point_lat_index[0] < (size_x_lat - 1000) or \
#             #     project_point_lat_index[1] < 1400:
#             #     continue
#
#             # if project_point_lat_index[0] >= (size_x_lat - 700) or \
#             #     project_point_lat_index[1] >= 2984 or \
#             #     project_point_lat_index[0] < (size_x_lat - 1300) or \
#             #     project_point_lat_index[1] < 2100:
#             #     continue
#
#             if project_point_lat_index[0] >= (size_x_lat - 650) or \
#                 project_point_lat_index[1] >= 2460 or \
#                 project_point_lat_index[0] < (size_x_lat - 1100) or \
#                 project_point_lat_index[1] < 2200:
#                 continue
#
#             # if project_point_lat_index[0] >= size_x_lat or \
#             #     project_point_lat_index[1] >= size_y_lat or \
#             #     project_point_lat_index[0] < 0 or \
#             #     project_point_lat_index[1] < 0:
#             #     continue
#
#             e_ap = project_point_ap - source_ap
#             n_ap = np.array([0, -1, 0])
#             x_ray_ap_grad = x_ray_ap_grad_array[project_point_ap_index[1]][project_point_ap_index[0]]
#             vector_ap = (np.cross(np.cross(n_ap, np.array([x_ray_ap_grad[0], 0, -x_ray_ap_grad[1]])), e_ap) / (np.dot(n_ap, e_ap))) * \
#                 (np.linalg.norm(project_point_ap - source_ap) / np.linalg.norm(reconstruct_point - source_ap))
#
#             e_lat = project_point_lat - source_lat
#             n_lat = np.array([1, 0, 0])
#             x_ray_lat_grad = x_ray_lat_grad_array[project_point_lat_index[1]][project_point_lat_index[0]]
#             vector_lat = (np.cross(np.cross(n_lat, np.array([0, x_ray_lat_grad[0], -x_ray_lat_grad[1]])), e_lat) / (np.dot(n_lat, e_lat))) * \
#                 (np.linalg.norm(project_point_lat - source_lat) / np.linalg.norm(reconstruct_point - source_lat))
#
#             vector_final = vector_ap + vector_lat
#
#             vector_field[k][j][i][0] = vector_final[0]
#             vector_field[k][j][i][1] = vector_final[1]
#             vector_field[k][j][i][2] = vector_final[2]
#
# np.save("vector_field_part.npy", vector_field)
vector_field = np.load("vector_field.npy")

# vector_field = CT_grad_array.copy()

# Show vector field
vector_field_norm = np.linalg.norm(vector_field, axis=3)
vector_field_mean = np.mean(vector_field_norm, 1)
vector_field_mean_copy = vector_field_mean.copy()
vector_field_mean_copy[vector_field_mean_copy == np.min(vector_field_mean_copy)] = np.max(vector_field_mean_copy)
second_min = np.min(vector_field_mean_copy)
vector_field_mean[vector_field_mean == np.min(vector_field_mean)] = second_min
vector_field_mean_ud = np.mean(vector_field_norm, 0)
vector_field_mean_ap = np.mean(vector_field_norm, 1)
vector_field_mean_lat = np.mean(vector_field_norm, 2)
plt.imshow(vector_field_mean, cmap='gray')
plt.show()

# Choose strong intensity gradients
elliptical_mask = np.load("elliptical_mask_for_grad_rec.npy")
CT_grad_array[elliptical_mask == False] = np.array([0, 0, 0])
CT_grad_norm = np.linalg.norm(CT_grad_array, axis=3)
strong_intensity_mask = CT_grad_norm > 500  # strong_intensity_mask = CT_grad_norm > np.median(CT_grad_norm)
strong_intensity_mask_index = np.array(np.where(strong_intensity_mask == True)).T
CT_grad_strong = CT_grad_array[strong_intensity_mask]
CT_grad_strong_unit = CT_grad_strong / np.tile(np.linalg.norm(CT_grad_strong, axis=1), (3, 1)).T
# CT_grad_strong_unit2 = CT_grad_array[strong_intensity_mask] / np.tile(CT_grad_norm[strong_intensity_mask], (3, 1)).T
# print((CT_grad_strong_unit == CT_grad_strong_unit2).all())
CT_in_strong = CT_in_cartesian_array[strong_intensity_mask]
print(CT_in_cartesian_array[0][0][0])

# Show CT gradient field
# print(np.max(CT_grad_array), np.min(CT_grad_array))
# CT_grad_norm[strong_intensity_mask] = -997
CT_grad_array[strong_intensity_mask == False] = np.array([0, 0, 0])
CT_grad_norm = np.linalg.norm(CT_grad_array, axis=3)
CT_grad_mean_ud = np.mean(CT_grad_norm, 0)
CT_grad_mean_ap = np.mean(CT_grad_norm, 1)
CT_grad_mean_lat = np.mean(CT_grad_norm, 2)
plt.imshow(np.squeeze(CT_grad_mean_lat))
plt.show()

# Calculate centroid
vector_field_part = np.load("vector_field_part.npy")
vector_field_part_norm = np.squeeze(np.linalg.norm(vector_field_part, axis=3))
vector_field_part_positive_norm_position = CT_in_cartesian_array[vector_field_part_norm > 0]
vector_field_part_position_centroid = np.mean(vector_field_part_positive_norm_position, axis=0)
CT_grad_positive_norm_position = CT_in_cartesian_array[strong_intensity_mask]
CT_grad_position_centroid = np.mean(CT_grad_positive_norm_position, axis=0)

# Initialization
T_start = np.identity(4)
R_start = np.identity(3)
t_start = vector_field_part_position_centroid - CT_grad_position_centroid
T_start[0:3, 0:3] = R_start
T_start[0:3, 3] = t_start.T
CT_in_strong_minus_center = CT_in_strong - np.tile(center, (CT_in_strong.shape[0], 1))
CT_in_strong_start = (np.dot(R_start, CT_in_strong_minus_center.T) + np.tile(t_start, (CT_in_strong_minus_center.shape[0], 1)).T).T + np.tile(center, (CT_in_strong.shape[0], 1)) # positon
CT_grad_strong_start = np.dot(R_start, CT_grad_strong.T).T  # gradient
CT_in_strong = np.array(CT_in_strong_start)
CT_grad_strong = np.array(CT_grad_strong_start)
CT_grad_strong_unit = np.array(CT_grad_strong_start / np.tile(np.linalg.norm(CT_grad_strong_start, axis=1), (3, 1)).T)
print("T_start:", T_start)

# Iteration
T = T_start
for iteration_k in range(1, 300):
    # Hypothesis generation. Use point-direction linear function.
    sigma_d = 10
    p_cr = sigma_d / np.sqrt(iteration_k)
    line_factor_lower = -2 * p_cr  # suppose to be: line_factor_lower = -2 * p_cr / np.linalg.norm(CT_grad_strong_unit, axis=1)
    CT_in_strong_next = np.array(CT_in_strong)
    p2p_similarity_best = np.zeros(CT_in_strong.shape[0])
    for line_factor in np.arange(line_factor_lower, -line_factor_lower, -line_factor_lower / 20):
        for x in range(CT_in_strong.shape[0]):
            CT_in_strong_middle = CT_in_strong[x] + line_factor * CT_grad_strong_unit[x]
            # index of CT strong points
            # CT_in_strong_middle_index_to_inverse = (CT_in_strong[x] - CT_in_cartesian_array[0][0][0]) / np.array([spacing_x, spacing_y, spacing_z])
            # CT_in_strong_middle_index_inversed = CT_in_strong_middle_index_to_inverse[::-1]
            # CT_in_strong_middle_index_inversed = np.array(CT_in_strong_middle_index_inversed, dtype=np.int)
            # index of vector field strong points
            vector_field_strong_middle_index_to_inverse = (CT_in_strong_middle - CT_in_cartesian_array[0][0][0]) / np.array([spacing_x, spacing_y, spacing_z])
            vector_field_strong_middle_index_inversed = vector_field_strong_middle_index_to_inverse[::-1]
            vector_field_strong_middle_index_inversed = np.array(vector_field_strong_middle_index_inversed, dtype=np.int)
            if (0 < vector_field_strong_middle_index_inversed[0] and vector_field_strong_middle_index_inversed[0] < size_z and \
                0 < vector_field_strong_middle_index_inversed[1] and vector_field_strong_middle_index_inversed[1] < size_y and \
                0 < vector_field_strong_middle_index_inversed[2] and vector_field_strong_middle_index_inversed[2] < size_x):

                # index1 = CT_in_strong_middle_index_inversed
                index2 = vector_field_strong_middle_index_inversed
                CT_grad_strong_middle = CT_grad_strong[x]
                vector_field_strong_middle = vector_field[index2[0]][index2[1]][index2[2]]

                if np.linalg.norm(CT_grad_strong_middle) == 0 or \
                    np.linalg.norm(vector_field_strong_middle) == 0:
                    continue

                f_alpha = np.sum(CT_grad_strong_middle * vector_field_strong_middle) / \
                    (np.linalg.norm(CT_grad_strong_middle) * np.linalg.norm(vector_field_strong_middle))

                if f_alpha < 0:
                    f_alpha = 0

                g_d = np.exp(-0.5 * ((line_factor / p_cr) ** 2))

                p2p_similarity = np.linalg.norm(CT_grad_strong_middle) * \
                                np.linalg.norm(vector_field_strong_middle) * \
                                (f_alpha ** 4) * g_d

                if p2p_similarity > p2p_similarity_best[x]:
                    CT_in_strong_next[x] = CT_in_strong_middle
                    p2p_similarity_best[x] = p2p_similarity
                    line_factor_max = line_factor
                    # print(line_factor)
        
    # print(np.linalg.norm(CT_in_strong_next - CT_in_strong))
    # print("line_factor_max:", line_factor_max)
    # print("p2p_similarity_best:", p2p_similarity_best)

    # Calculate hypothetical incremental transformation using RANSAC
    CF_max = 0  # criterion function
    R_best = np.identity(3)
    t_best = np.zeros(3)
    for i in range(100):
        # Randomly sample three points to calculate rigid motion matrix
        pointset_size = 20
        new_pointset_index = random.sample(range(CT_in_strong.shape[0]), pointset_size)
        pointset = CT_in_strong[new_pointset_index] - np.tile(center, (pointset_size, 1))
        pointset_next = CT_in_strong_next[new_pointset_index] - np.tile(center, (pointset_size, 1))

        # test
        # R1 = np.random.rand(3, 3)
        # t1 = np.random.rand(3, 1)
        # U1, S1, Vt1 = np.linalg.svd(R1)
        # R1 = np.dot(U1, Vt1)
        # if np.linalg.det(R1) < 0:
        #         print("Reflection detected")
        #         Vt1[2, :] *= -1
        #         R1 = np.dot(Vt1.T, U1.T)
        # pointset = np.random.rand(pointset_size, 3)
        # pointset_next = (np.dot(R1, pointset.T) + np.tile(t1, (1, pointset_size))).T
        # print(R1, t1)

        # Decentralization
        pointset_decentralized = pointset - np.tile(np.mean(pointset, axis=0), (pointset_size, 1))
        pointset_next_decentralized = pointset_next - np.tile(np.mean(pointset_next, axis=0), (pointset_size, 1))

        # Calculate covariance matrix
        H = np.dot(pointset_decentralized.T, pointset_next_decentralized)

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Calculate rotation
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            # print("Reflection detected")
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Calculate translation
        t = np.mean(pointset_next, axis=0) - np.dot(R, np.mean(pointset, axis=0))
        # print(R, t)

        # Apply transform matrix
        CT_in_strong_minus_center = CT_in_strong - np.tile(center, (CT_in_strong.shape[0], 1))
        CT_in_strong_next_pred = (np.dot(R, CT_in_strong_minus_center.T) + np.tile(t, (CT_in_strong_minus_center.shape[0], 1)).T).T + np.tile(center, (CT_in_strong.shape[0], 1)) # positon
        CT_grad_strong_next_pred = np.dot(R, CT_grad_strong.T).T  # gradient
        CF_1 = 0  # criterion function
        CF_2 = 0
        CF_3 = 0
        for x in range(CT_in_strong_next_pred.shape[0]):
            vector_field_next_pred_position = CT_in_strong_next_pred[x]
            vector_field_next_pred_index = (vector_field_next_pred_position - CT_in_cartesian_array[0][0][0]) / np.array([spacing_x, spacing_y, spacing_z])
            vector_field_next_pred_index = vector_field_next_pred_index[::-1]
            vector_field_next_pred_index = np.array(vector_field_next_pred_index, dtype=np.int)
            if (0 < vector_field_next_pred_index[0] and vector_field_next_pred_index[0] < size_z and \
                0 < vector_field_next_pred_index[1] and vector_field_next_pred_index[1] < size_y and \
                0 < vector_field_next_pred_index[2] and vector_field_next_pred_index[2] < size_x):
                vector_field_next_pred_grad = vector_field[vector_field_next_pred_index[0]][vector_field_next_pred_index[1]][vector_field_next_pred_index[2]]
                Tup = np.linalg.norm(CT_grad_strong_next_pred[x])
                vTp = np.linalg.norm(vector_field_next_pred_grad)
                if Tup <= 1 or vTp <= 1:
                    continue
                f_alpha = np.sum(CT_grad_strong_next_pred[x] * vector_field_next_pred_grad) / (Tup * vTp)
                if f_alpha < 0:
                    f_alpha = 0
                CF_1 += Tup * vTp * f_alpha
                CF_2 += Tup
                CF_3 += vTp
        CF = CF_1 / (CF_2 * CF_3)
        print(CF)

        # Save best transform matrix
        if CF > CF_max:
            CF_max = CF
            R_best = R
            t_best = t
            print("max:", CF_max, R_best, t_best)
    
    # Apply best transform matrix
    CT_in_strong_minus_center = CT_in_strong - np.tile(center, (CT_in_strong.shape[0], 1))
    CT_in_strong_next_pred_best = (np.dot(R_best, CT_in_strong_minus_center.T) + np.tile(t_best, (CT_in_strong_minus_center.shape[0], 1)).T).T + np.tile(center, (CT_in_strong.shape[0], 1)) # positon
    CT_grad_strong_next_pred_best = np.dot(R_best, CT_grad_strong.T).T  # gradient
    
    # Update CT strong points and gradient
    CT_in_strong = np.array(CT_in_strong_next_pred_best)
    CT_grad_strong = np.array(CT_grad_strong_next_pred_best)
    CT_grad_strong_unit = np.array(CT_grad_strong_next_pred_best / np.tile(np.linalg.norm(CT_grad_strong_next_pred_best, axis=1), (3, 1)).T)
    
    # Show result
    CT_grad_norm_show = np.zeros((size_z, size_y, size_x))
    for x in range(CT_in_strong.shape[0]):
            vector_field_strong_index_to_inverse = (CT_in_strong[x] - CT_in_cartesian_array[0][0][0]) / np.array([spacing_x, spacing_y, spacing_z])
            vector_field_strong_index_inversed = vector_field_strong_index_to_inverse[::-1]
            vector_field_strong_index_inversed = np.array(vector_field_strong_index_inversed, dtype=np.int)
            if (0 < vector_field_strong_index_inversed[0] and vector_field_strong_index_inversed[0] < size_z and \
                0 < vector_field_strong_index_inversed[1] and vector_field_strong_index_inversed[1] < size_y and \
                0 < vector_field_strong_index_inversed[2] and vector_field_strong_index_inversed[2] < size_x):
                index = vector_field_strong_index_inversed
                CT_grad_norm_show[index[0]][index[1]][index[2]] = np.linalg.norm(CT_grad_strong[x])
    plt.subplot(321)
    plt.imshow(np.squeeze(vector_field_mean_ud))
    plt.subplot(322)
    plt.imshow(np.squeeze(np.mean(CT_grad_norm_show, axis=0)))
    plt.subplot(323)
    plt.imshow(np.squeeze(vector_field_mean_ap))
    plt.subplot(324)
    plt.imshow(np.squeeze(np.mean(CT_grad_norm_show, axis=1)))
    plt.subplot(325)
    plt.imshow(np.squeeze(vector_field_mean_lat))
    plt.subplot(326)
    plt.imshow(np.squeeze(np.mean(CT_grad_norm_show, axis=2)))
    plt.show()

    T_delta = np.concatenate((np.concatenate((R_best, t_best.reshape(3, 1)), axis=1), np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    T = np.dot(T_delta, T)
    print(T)
    print("Max:", R_best, t_best, CF_max)