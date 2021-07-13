import os
from zipfile import ZipFile

def unzip_files(directory):
    """
    This function searches for zip files in a given directory, unzips them and deletes them.

    @params:
        directory (str): The directory to search through
    """
    # Using the os package to change to the desired directory
    current_directory = os.getcwd()
    os.chdir(directory)

    # Extracting zip files
    for file_name in os.listdir():
        if '.zip' == file_name[-4:]:
            with ZipFile(file_name, mode='r') as zip:
                print(f"Unzipping {file_name}")
                zip.extractall()
            os.remove(file_name)
            print(f"Deleted {file_name}")
    
    # changing current directory back to the original directory
    os.chdir(current_directory)

def get_kaggle_path(kaggle_link):
    """
    This functin grabs the path of the dataset from the kaggle url of the dataset

    ex.
        Link: https://www.kaggle.com/mikhailgaerlan/spongebob-squarepants-completed-transcripts
        Path: mikhailgaerlan/spongebob-squarepants-completed-transcripts

    @params:
        kaggle_link (str): The url of the kaggle dataset

    @return:
        kaggle_path (str): The path of the dataset
    """

    return ('/').join(kaggle_link.split('/')[3:])


def download_dataset(kaggle_link, api):
    """
    This function accepts the kaggle dataset link and Kaggle's API token to download dataset from Kaggle.

    @params:
        kaggle_link (str): The url of the kaggle dataset
        api (str): The Kaggle API token
    """

    # Downloading Dataset from Kaggle
    kaggle_path = get_kaggle_path(kaggle_link)
    api.dataset_download_files(kaggle_path)
    
    # Unzipping the files in the current path
    unzip_files(os.getcwd())