"""
Script to check files for null bytes and report corrupted files
"""
import os
import sys

def check_file_for_null_bytes(file_path):
    """Check if a file contains null bytes"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                return True
        return False
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

def scan_directory_for_null_bytes(directory):
    """Scan a directory for files with null bytes"""
    corrupted_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if check_file_for_null_bytes(file_path):
                    corrupted_files.append(file_path)
    
    return corrupted_files

if __name__ == "__main__":
    # Directory to scan - defaults to current directory if not specified
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print(f"Scanning directory {scan_dir} for Python files with null bytes...")
    corrupted_files = scan_directory_for_null_bytes(scan_dir)
    
    if corrupted_files:
        print(f"Found {len(corrupted_files)} files with null bytes:")
        for file in corrupted_files:
            print(f"  - {file}")
    else:
        print("No files with null bytes found.")
