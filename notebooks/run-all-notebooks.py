import os
import subprocess
import time

# Get current timestamp
current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

# Create log directory with current timestamp
log_dir = os.path.join(".", "log", current_time)
os.makedirs(log_dir)

# Find all .ipynb files recursively from current directory
notebook_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb"):
            notebook_files.append(os.path.join(root, file))

# Run jupyter-runner for each notebook file found
for file in notebook_files:
    cmd = ["jupyter-runner", "--allow-errors", file, "--output-directory", log_dir, "--debug"]
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
