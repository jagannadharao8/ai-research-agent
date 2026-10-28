"""Legacy helper disabled until a genuinely isolated execution service exists."""

def extract_python_code(text):
    import re
    matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    return "\n".join(matches) if matches else text

def run_python_code(code, dataset_path=None, dataset_name=None):
    raise RuntimeError("Arbitrary code execution is disabled. Use a separately isolated execution service.")
