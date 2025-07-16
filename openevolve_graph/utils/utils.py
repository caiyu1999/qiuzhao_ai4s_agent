
def load_initial_program(path:str)->str:
    with open(path,"r") as f:
        return f.read()
    

    
    
    
    
    
    
    
if __name__ == "__main__": 
    print(load_initial_program("/Users/caiyu/Desktop/langchain/new_openevolve/examples/circle_packing/initial_program.py"))