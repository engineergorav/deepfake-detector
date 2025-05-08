# utils.py

# Example function for checking if the file is an image
def is_valid_image(file):
    return file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
