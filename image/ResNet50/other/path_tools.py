import inspect
import os
import sys

# path tools provided by Alex


def get_solution_directory():
    # Assumes that all scripts that are actually executed reside in a sub-folder of AssignmentX,
    # e.g. Assignment2/TestScripts, therefore returns the parent of the currently executed file
    solution_directory = os.path.realpath(
        os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
    return solution_directory


def get_dataset_directory():
    return os.path.join(get_solution_directory(), "..", "Datasets")


def get_cifar10_directory():
    return os.path.join(get_dataset_directory(), "cifar-10-batches-py")


def set_execution_environment_path():
    cmd_parent = get_solution_directory()
    sys.path.append(cmd_parent)
    print("Appending parent-path: {0}".format(cmd_parent))


def get_cifar10_hog_directory():
    return os.path.join(get_dataset_directory(), "cifar-10-hog-SVM_image_features")