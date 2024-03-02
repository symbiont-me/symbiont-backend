# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime


def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)


def replace_space_with_underscore(text):
    return text.replace(" ", "_")


def make_file_identifier(text):
    """
    Generate a file identifier by cleaning the given text and appending the current datetime.

    Args:
        text (str): The input text to generate the file identifier from.

    Returns:
        str: The generated file identifier.
    """
    cleaned_filename = remove_non_ascii(replace_space_with_underscore(text))
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    identifier = f"{current_datetime}_{cleaned_filename}"
    return identifier
