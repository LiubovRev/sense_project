import json

def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def escape_glob_path(path):
    """
    Escapes special characters from a glob path.
    Special glob characters: '*', '?', '[', ']', etc.

    Args:
    path (str): The file path to be escaped for use with glob.

    Returns:
    str: The escaped file path.
    """
    # List of special glob characters that need to be escaped
    special_chars = r'*?[]'
    # Using a raw string to avoid issues with backslashes
    escaped_path = re.sub(r'([' + re.escape(special_chars) + r'])', r'[\1]', path)
    return escaped_path
