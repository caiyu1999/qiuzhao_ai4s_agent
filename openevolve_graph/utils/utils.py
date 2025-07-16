import os 
import re 
def load_initial_program(path:str)->str:
    # 检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    with open(path,"r") as f:
        return f.read()
    
    

def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"
    
    
    
    
    
    
if __name__ == "__main__": 
    print(load_initial_program("/Users/caiyu/Desktop/langchain/new_openevolve/examples/circle_packing/initial_program.py"))