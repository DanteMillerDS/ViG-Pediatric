import shutil
from google.colab import drive
import os

def mount_google_drive():
    """
    Mounts Google Drive to the Colab environment if it is not already mounted.
    :return: None.
    """
    
    print("Mounting Google Drive...")
    drive.mount('/content/drive')


def copy_files_from_drive():
    """
    Copies specified files from Google Drive to the Colab local workspace.
    :return: None. Files are copied to the Colab local workspace.
    """
    print("Copying files from Google Drive...")
    shutil.copy("/content/drive/MyDrive/ucsd.zip", "/content/ucsd.zip")
    shutil.copy("/content/drive/MyDrive/ori.zip", "/content/ori.zip")

def extract_zip_files():
    """
    Extracts the contents of specified zip files into the local workspace.
    :return: None. Files are extracted to the specified output directory within the local workspace.
    """
    print("Extracting zip files...")
    get_ipython().system('7z x /content/ucsd.zip -o/content/')
    get_ipython().system('7z x /content/ori.zip -o/content/')
    print("Files extracted successfully.")

def mount_and_process():
    """
    Coordinates the mounting of Google Drive, copying of files from Google Drive to the local workspace,
    and extracting those files for processing. 
    :return: None.
    """
    if not os.path.ismount('/content/drive'):
        mount_google_drive()
        copy_files_from_drive()
        extract_zip_files()
    else:
        print("Google Drive is already mounted.")