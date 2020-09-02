import os

def safe_remove(file_path):
    """
    Remove a file after checking that it exists

    Args:
        file_path (str): File path that should be removed

    Return:
        None
    """
    if os.path.exists(file_path):
        os.remove(file_path)
