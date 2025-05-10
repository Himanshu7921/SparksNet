import os

def generate_file_structure(startpath, output_file="project_structure.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(startpath):
            # Skip .git and __pycache__ directories
            if '.git' in root or '__pycache__' in root:
                continue

            level = root.replace(startpath, "").count(os.sep)
            indent = "│   " * level + "├── " if level else ""
            f.write(f"{indent}{os.path.basename(root)}/\n")

            sub_indent = "│   " * (level + 1) + "├── "
            for file in files:
                # Skip git-related files (like .gitignore, etc.)
                if file.startswith(".git") or file == "__init__.pyc":
                    continue
                f.write(f"{sub_indent}{file}\n")
    
    print(f"✅ File structure saved to {output_file}")

# Run the function in the current directory
generate_file_structure(os.getcwd())
