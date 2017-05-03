import pickle

from keras.models import load_model

from data_handling_and_preparation import SAP
import data_handling_and_preparation as dhap

import working_directory_definition as wdd

def detect_sea_lions_in_image(filename):
    train_image = cv2.imread(train_image_filename)

    image_patches_list = slice_the_image_into_patches(train_image, patch_h, patch_w)




if __name__ == '__main__':

    directories = wdd.check_directory_structure_trainsmall2()
    top_dir = directories["TOP_DIR"]
    version_directory = dhap.get_current_version_directory(top_dir)

    parameter_file = open(version_directory + "parameters_file.pkls", "rb")
    parameters = pickle.load(parameter_file)

    print(parameters)

    #detection_model_hdf5_filename = "detection_model.h5"
    #counting_model_hdf5_filename = "counting_model.h5"


    #detection_model = load_model(detection_model_hdf5_filename)
    #counting_model = load_model(counting_model_hdf5_filename)


