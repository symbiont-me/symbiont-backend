# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
import re


def replace_non_alphanumeric(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text)


def make_file_identifier(filename):
    """
    Generate a file identifier by cleaning the given text and appending the current datetime.

    Args:
        text (str): The input text to generate the file identifier from.

    Returns:
        str: The generated file identifier.
    """
    cleaned_filename = replace_non_alphanumeric(filename)
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    identifier = f"{current_datetime}_{cleaned_filename}"
    return identifier
