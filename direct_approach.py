import os
import sys
import pickle

from data_handling_and_preparation import SAP
import data_handling_and_preparation as dhap


if __name__ == "__main__":
    print("Simple direct approach.")

    top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    directories = dhap.get_current_version_directory(top_dir)


    data_directory = directories["DATA_DIRECTORY"]
    train_images_dir = directories["TRAIN_DATA_DIRECTORY"]
    train_dotted_images_dir = directories["TRAIN_DOTTED_DATA_DIRECTORY"]

    preprocessed_detection_data_dir = directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"]
    preprocessed_counting_data_dir = directories["PREPROCESSED_COUNTING_DATA_DIRECTORY"]
    train_csv_filename = train_images_dir + "train.csv"
    mismatched_train_images_filename = top_dir + "MismatchedTrainImages.txt"

    invalid_images_list = dhap.read_csv(mismatched_train_images_filename)
    invalid_images_list = [invalid_images_list[i][0] for i in range(len(invalid_images_list))]

    # These are the parameters that will be used to generate the data.
    # They are generated with one of the data_generation_parameters_ver_*.py scripts
    # Generally the path should be top_dir + "Parameters_and_models_ver_x/parameters_file.pkls"
    # where x is the version of the parameters used.
    detection_parameters_filename = directories["PARAMETERS_FILENAME"]
    
    # Check if files exists
    if (os.path.isfile(detection_parameters_filename) == False):
        sys.exit("ERROR: The parameter files does not exists.")

    print("Directories that will be used:")
    print("top_dir: %s" % (top_dir))
    print("data_directory: %s" % (data_directory))
    print("train_images_dir: %s" % (train_images_dir))
    print("train_dotted_images_dir: %s" % (train_dotted_images_dir))
    print("preprocessed_detection_data_dir: %s" % (preprocessed_detection_data_dir))
    print("preprocessed_counting_data_dir: %s" % (preprocessed_counting_data_dir))
    print("train_csv_filename: %s" % (train_csv_filename))
    print("detection_parameters_filename: %s\n" % (detection_parameters_filename))

    expected_lion_count_list = dhap.read_csv(train_csv_filename)


    train_image_filename_list = dhap.get_filename_list_in_dir(train_images_dir, file_type="jpg")
    train_dotted_image_filename_list = dhap.get_filename_list_in_dir(train_dotted_images_dir, file_type="jpg")

    if (len(train_image_filename_list) != len(train_dotted_image_filename_list)):
        sys.exit("ERROR: Filename lists have different lengths.")

    print("Files that will be processed:")
    for i in range(len(train_image_filename_list)):    
        print(train_image_filename_list[i])
        print(train_dotted_image_filename_list[i])
    print()


    """
    dhap.direct_approach_full_input_image(train_image_filename_list[0], 
                                     train_dotted_image_filename_list[0],
                                     patch_h=500,
                                     patch_w=500,
                                     resize_image_patch_to_h=256,
                                     resize_image_patch_to_w=256,
                                     radious_list = [32, 32, 32, 16, 32],
                                     sap_list=[SAP(0.0, 1.0), SAP(90.0, 1.0)],
                                     interactive_plot=True)
    """

    parameters = {}
    parameter_file = open(detection_parameters_filename, "rb")

    parameters = pickle.load(parameter_file)

    # Detection data parameters for dispatch.
    patch_h = parameters["patch_h"]
    patch_w = parameters["patch_w"]
    resize_image_patch_to_h = parameters["resize_image_patch_to_h"]
    resize_image_patch_to_w = parameters["resize_image_patch_to_w"]
    sap_list = parameters["sap_list"]


    dhap.prepare_and_dispatch_lion_direct_approach_data(train_image_filename_list, 
                                                        train_dotted_image_filename_list,
                                                        preprocessed_detection_data_dir, # data are saved in the detection dir
                                                        invalid_images_list,
                                                        patch_h=patch_h,
                                                        patch_w=patch_w,
                                                        resize_image_patch_to_h=resize_image_patch_to_h,
                                                        resize_image_patch_to_w=resize_image_patch_to_w,
                                                        sap_list=sap_list)

    """
    prepare_and_dispatch_lion_detection_data(train_image_filename_list, 
                                             train_dotted_image_filename_list,
                                             preprocessed_detection_data_dir,
                                             invalid_images_list,
                                             patch_h=patch_h,
                                             patch_w=patch_w,
                                             resize_image_patch_to_h=resize_image_patch_to_h,
                                             resize_image_patch_to_w=resize_image_patch_to_w,
                                             resize_mask_patch_to_h=resize_mask_patch_to_h,
                                             resize_mask_patch_to_w=resize_mask_patch_to_w,
                                             radious_list=radious_list,
                                             sap_list=sap_list,
                                             interactive_plot=interactive_plot,
                                             display_every=display_every)
    """


