import os
import subprocess
import re
import tempfile
import uuid

def extract_python_code(text):
    """Extracts python code blocks from markdown text."""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return text

def run_python_code(code, dataset_path=None, dataset_name=None):
    """
    Runs python code in a temporary directory and captures stdout and any generated images.
    Returns (stdout, image_paths).
    """
    code = extract_python_code(code)
    
    # Create a temporary directory to store the script and its outputs
    temp_dir = tempfile.mkdtemp()
    
    if dataset_path and dataset_name:
        import shutil
        shutil.copy(dataset_path, os.path.join(temp_dir, dataset_name))
        
    script_path = os.path.join(temp_dir, "sandbox.py")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    try:
        # Run the script with a 15-second timeout
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=temp_dir
        )
        
        output = result.stdout
        if result.stderr:
            output += "\nERROR:\n" + result.stderr
            
    except subprocess.TimeoutExpired:
        output = "Execution Timed Out (15 seconds max)."
    except Exception as e:
        output = f"Execution Error: {str(e)}"
        
    # Look for any image files generated in the directory
    image_paths = []
    for file in os.listdir(temp_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_paths.append(os.path.join(temp_dir, file))
            
    return output, image_paths
