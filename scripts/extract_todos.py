import os
import fnmatch


def extract_todos_from_file(file_path, root_dir):
    todos = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if line.startswith("# TODO") or line.startswith("# @todo"):
                    # Calculate the relative path
                    relative_path = os.path.relpath(file_path, root_dir)
                    todos.append((relative_path, line_number, line))
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()
                    if line.startswith("# TODO") or line.startswith("# @todo"):
                        # Calculate the relative path
                        relative_path = os.path.relpath(file_path, root_dir)
                        todos.append((relative_path, line_number, line))
        except UnicodeDecodeError:
            print(f"Skipping file due to encoding error: {file_path}")
    return todos


def extract_todos_from_directory(directory, exclude_patterns=None):
    todos = []
    exclude_patterns = exclude_patterns or []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories based on pattern
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)]

        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                todos.extend(extract_todos_from_file(file_path, directory))
    return todos


def read_existing_todos(output_file):
    existing_todos = set()
    try:
        with open(output_file, "r", encoding="utf-8") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith("- Line "):
                    existing_todos.add(stripped_line)
    except FileNotFoundError:
        # If the file does not exist, we just return an empty set
        pass
    return existing_todos


def write_todos_to_markdown(todos, output_file, existing_todos):
    try:
        with open(output_file, "a", encoding="utf-8") as file:
            for file_path, line_number, comment in todos:
                todo_line = f"- Line {line_number}: {comment}"
                if todo_line not in existing_todos:
                    file.write(f"## {file_path}\n")
                    file.write(f"{todo_line}\n\n")
    except Exception as e:
        print(f"An error occurred while writing: {e}")


if __name__ == "__main__":
    # Root directory to begin search from (one directory up from the current script)
    directory = os.path.join(os.path.dirname(__file__), "..")  # Project root directory
    print(f"Starting search in directory: {directory}")

    # List of patterns to exclude
    exclude_patterns = [
        "__pycache__",
        "*.egg-info",
        ".git",
        # Add more patterns as needed
    ]

    # Extract TODOs
    todos = extract_todos_from_directory(directory, exclude_patterns)

    # Define the output markdown file path
    output_file = "todos.md"

    # Read existing TODOs from the markdown file
    existing_todos = read_existing_todos(output_file)

    # Write TODOs to the markdown file, excluding existing ones
    write_todos_to_markdown(todos, output_file, existing_todos)

    print(f"Extracted TODO comments have been written to {output_file}")
