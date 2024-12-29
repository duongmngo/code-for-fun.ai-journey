import os

def get_absolute_path(current_dir, relative_path):
    return os.path.abspath(os.path.join(current_dir, relative_path))