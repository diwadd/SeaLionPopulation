import pickle

from data_handling_and_preparation import SAP
import data_handling_and_preparation as dhap
import working_directory_definition as wdd

# Data generation parameters.
# Version x

directories = wdd.check_directory_structure_trainsmall2()
top_dir = directories["TOP_DIR"]

version = "x"
detection_parameters_directory = dhap.check_if_dir_exists_create_it_if_not(top_dir + "Parameters_and_models_ver_" + version + "/")
detection_parameters_filename = detection_parameters_directory + "parameters_file.pkls"

f = open(top_dir + "current_version", "w")
f.write(detection_parameters_directory)
f.close()

parameters = {}
parameter_file = open(detection_parameters_filename, "wb")

parameters["version"] = version
parameters["version_directory"] = detection_parameters_directory

# Detection data parameters for dispatch.
patch_h=500
patch_w=500
resize_image_patch_to_h=128
resize_image_patch_to_w=128
resize_mask_patch_to_h=32
resize_mask_patch_to_w=32
radious_list=[20, 20, 20, 12, 20]
sap_list=[SAP(0.0, 1.0)]
interactive_plot=False
display_every=10

parameters["patch_h"] = patch_h
parameters["patch_w"] = patch_w
parameters["resize_image_patch_to_h"] = resize_image_patch_to_h
parameters["resize_image_patch_to_w"] = resize_image_patch_to_w
parameters["resize_mask_patch_to_h"] = resize_mask_patch_to_h
parameters["resize_mask_patch_to_w"] = resize_mask_patch_to_w
parameters["radious_list"] = radious_list
parameters["sap_list"] = sap_list

# Counting data parameters for dispatch.
counting_radious=15
nh=32 # final image size - height
nw=32 # final image size - width
counting_dot_threshold=1
lions_contour_dot_threshold=1
h_threshold=16 # minimal height (size) of a single lion that will be cropped
w_threshold=16 # minimal width (size) of a single lion that will be cropped
rectangle_shape=True

# Redefine for lion images
interactive_plot=False
display_every=1

parameters["counting_radious"] = counting_radious
parameters["nh"] = nh
parameters["nw"] = nw
parameters["counting_dot_threshold"] = counting_dot_threshold
parameters["lions_contour_dot_threshold"] = lions_contour_dot_threshold
parameters["h_threshold"] = h_threshold
parameters["w_threshold"] = w_threshold
parameters["rectangle_shape"] = rectangle_shape

pickle.dump(parameters, parameter_file)
parameter_file.close()

