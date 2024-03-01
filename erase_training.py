# stole this from https://www.tutorialspoint.com/How-to-delete-all-files-in-a-directory-with-Python
import os

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

# Usage
folds = ['test', 'train']
dirs = ['AR', 'EN', 'FR', 'JP']
for fold in folds:
    for dir in dirs:
        directory_path = 'melspecs/' + fold + '/' + dir
        delete_files_in_directory(directory_path)
