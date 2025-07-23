'''

这个代码用来测试子图的相关问题

'''
from langgraph.graph import START, END ,StateGraph

from pydantic import BaseModel,Field


class State(BaseModel):
    a:str = Field(default="a")
    b:str = Field(default="b")
    c:str = Field(default="c")
    
    
builder = StateGraph(State)
left_builder = StateGraph(State)
right_builder = StateGraph(State)

def node_init(state:State):
    '''
    什么也不更新 只是初始化
    '''
    return None 

def node_change_1_left(state:State):
    '''
    更新a
    '''
    print(f"左侧子图的node_change_1_left被调用前state: {state}")
    return {"a":"al"}

def node_change_1_right(state:State):
    '''
    更新a
    '''
    print(f"右侧子图的node_change_1_right被调用前state: {state}")
    return {"a":"ar"}


def node_change_2_left(state:State):
    '''
    更新b
    '''
    print(f"左侧子图的node_change_2_left被调用前state: {state}")
    return {"b":"bl"}

def node_change_2_right(state:State):
    '''
    更新b
    '''
    print(f"右侧子图的node_change_2_right被调用前state: {state}")
    return {"b":"br"}
    





#左侧子图
left_builder.add_node("init",node_init)
left_builder.add_node("change_1",node_change_1_left)
left_builder.add_node("change_2",node_change_2_left)
left_builder.add_edge(START,"init")
left_builder.add_edge("init","change_1")
left_builder.add_edge("change_1","change_2")
left_builder.add_edge("change_2",END)
left_graph = left_builder.compile()

#右侧子图
right_builder.add_node("init",node_init)
right_builder.add_node("change_1",node_change_1_right)
right_builder.add_node("change_2",node_change_2_right)
right_builder.add_edge(START,"init")
right_builder.add_edge("init","change_1")
right_builder.add_edge("change_1","change_2")
right_builder.add_edge("change_2",END)
right_graph = right_builder.compile()


#主图
builder.add_node("left",left_graph)
builder.add_node("right",right_graph)
builder.add_edge(START,"left")
builder.add_edge(START,"right")
builder.add_edge("left",END)
builder.add_edge("right",END)
graph = builder.compile()

state = State()
result = graph.invoke(state)
print(result)





