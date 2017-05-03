import os

import data_handling_and_preparation as dhap



def check_directory_structure_trainsmall2():

    print("Checking directory structure...")

    top_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

    directories = {}
    directories["TOP_DIR"] = top_dir
    directories["TRAINSMALL2_DATA_DIRECTORY"] = dhap.check_if_directory_exists(top_dir + "TrainSmall2/")
    directories["TRAIN_DATA_DIRECTORY"] = dhap.check_if_directory_exists(top_dir + "TrainSmall2/Train/")
    directories["TRAIN_DOTTED_DATA_DIRECTORY"] = dhap.check_if_directory_exists(top_dir + "TrainSmall2/TrainDotted/")

    directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"] = dhap.check_if_dir_exists_create_it_if_not(top_dir + "Detection_data/")
    directories["PREPROCESSED_COUNTING_DATA_DIRECTORY"] = dhap.check_if_dir_exists_create_it_if_not(top_dir + "Counting_data/")

    print("Directory structure is OK!")
    return directories


if __name__ == '__main__':

    directories = check_directory_structure_trainsmall2()

    print("TOP_DIRECTORY: %s" % (directories["TOP_DIR"]))
    print("TRAINSMALL2_DATA_DIRECTORY: %s" % (directories["TRAINSMALL2_DATA_DIRECTORY"]))
    print("TRAIN_DATA_DIRECTORY: %s" % (directories["TRAIN_DATA_DIRECTORY"]))
    print("TRAIN_DOTTED_DATA_DIRECTORY: %s" % (directories["TRAIN_DOTTED_DATA_DIRECTORY"]))
    print("PREPROCESSED_DETECTION_DATA_DIRECTORY: %s" % (directories["PREPROCESSED_DETECTION_DATA_DIRECTORY"]))
    print("PREPROCESSED_COUNTING_DATA_DIRECTORY: %s" % (directories["PREPROCESSED_COUNTING_DATA_DIRECTORY"]))








