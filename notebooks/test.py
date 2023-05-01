import os
import json
import nbformat
import datetime

def run_notebook(notebook_path):
    """Run a Jupyter notebook and report execution results"""

    # Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Set up a dictionary to hold results
    results = {
        "path": notebook_path,
        "cells": [],
        "execution_time": None,
        "num_errors": 0
    }

    # Execute each cell in the notebook
    start_time = datetime.datetime.now()
    for i, cell in enumerate(nb.cells):
        # Only execute code cells
        if cell.cell_type == "code":
            execution_result = {"cell_number": i, "execution_file": cell.execution_count}
            try:
                # Execute the code in the cell
                exec(cell.source)
                execution_result["success"] = True
            except Exception as e:
                # Record the error message
                execution_result["success"] = False
                execution_result["error_message"] = str(e)
                results["num_errors"] += 1

            # Add the cell results to the overall results
            results["cells"].append(execution_result)

    # Record the notebook execution time
    end_time = datetime.datetime.now()
    results["execution_time"] = (end_time - start_time).total_seconds()

    # Write the results to a log file
    log_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = f"./logs/log_{log_time}.json"
    with open(log_path, "w") as f:
        json.dump(results, f)

    # Return the results
    return results

def generate_html(log_path):
    with open(log_path) as f:
        logs = json.load(f)
    html_logs = []
    for log in logs:
        path = log['path']
        html_cells = []
        for cell in log['cells']:
            cell_number = cell['number']
            cell_status = cell['status']
            cell_execution_file = cell['execution_file']
            if cell_status == 'failed':
                cell_html = f"<li>Cell {cell_number} failed: {cell['error']}</li>"
            else:
                cell_html = f"<li>Cell {cell_number} successful</li>"
            html_cells.append(cell_html)
        html_logs.append(f"<div><h2>{path}</h2><ul>{''.join(html_cells)}</ul></div>")
    return f"<html><body>{''.join(html_logs)}</body></html>"

def run_tests(root_path):
    """Recursively run all Jupyter notebooks in a directory tree"""

def run_tests(root_path):
    log_path = f"./logs/log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    log_data = []
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        if os.path.isdir(file_path):
            log_data.extend(run_tests(file_path))
        elif file.endswith(".ipynb"):
            result = run_notebook(file_path)
            log_data.append(result)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    html = generate_html(log_path)
    with open("./logs/index.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    run_tests(".") # Replace "." with the root path of your notebook directory
