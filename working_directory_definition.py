import os
import json

import data_handling_and_preparation as dhap



def check_directory_structure_trainsmall2(data_dir):

    print("Checking directory structure...")

    top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

    directories = {}
    directories["TOP_DIR"] = top_dir
    directories["DATA_DIRECTORY"] = dhap.check_if_directory_exists(data_dir)
    directories["TRAIN_DATA_DIRECTORY"] = dhap.check_if_directory_exists(data_dir + "Train/")
    directories["TRAIN_DOTTED_DATA_DIRECTORY"] = dhap.check_if_directory_exists(data_dir + "TrainDotted/")

    directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"] = dhap.check_if_dir_exists_create_it_if_not(data_dir + "Detection_data/")
    directories["PREPROCESSED_COUNTING_DATA_DIRECTORY"] = dhap.check_if_dir_exists_create_it_if_not(data_dir + "Counting_data/")

    print("Directory structure is OK!")
    return directories


def save_current_version(data_dir, version):

    directories = check_directory_structure_trainsmall2(data_dir)
    top_dir = directories["TOP_DIR"]
    parameters_directory = dhap.check_if_dir_exists_create_it_if_not(top_dir + "Parameters_and_models_ver_" + version + "/")
    parameters_filename = parameters_directory + "parameters_file.pkls"


    directories["PARAMETERS_DIRECTORY"] = parameters_directory
    directories["PARAMETERS_FILENAME"] = parameters_filename

    # The current verion files defines all the folder that should be used.
    f = open(top_dir + "current_version", "w")
    json.dump(directories, f, sort_keys=True, indent=4, ensure_ascii=False)
    f.close()

    return directories


if __name__ == '__main__':

    directories = check_directory_structure_trainsmall2()

    print("TOP_DIRECTORY: %s" % (directories["TOP_DIR"]))
    print("TRAINSMALL2_DATA_DIRECTORY: %s" % (directories["TRAINSMALL2_DATA_DIRECTORY"]))
    print("TRAIN_DATA_DIRECTORY: %s" % (directories["TRAIN_DATA_DIRECTORY"]))
    print("TRAIN_DOTTED_DATA_DIRECTORY: %s" % (directories["TRAIN_DOTTED_DATA_DIRECTORY"]))
    print("PREPROCESSED_DETECTION_DATA_DIRECTORY: %s" % (directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"]))
    print("PREPROCESSED_COUNTING_DATA_DIRECTORY: %s" % (directories["PREPROCESSED_COUNTING_DATA_DIRECTORY"]))








