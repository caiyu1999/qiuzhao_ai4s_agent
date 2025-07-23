from typing import Annotated
import operator
from typing import TypedDict

from langgraph.types import Send
from langgraph.graph import END, START

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]



def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]


def generate_joke(state):
    print(state)
    return {"jokes": [f"Joke about {state['subject']}"]}

from langgraph.graph import StateGraph
builder = StateGraph(OverallState)
builder.add_node("generate_joke", generate_joke)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/Graph/graph_sample.png')

# Invoking with two subjects results in a generated joke for each
result = graph.invoke({"subjects": ["cats", "dogs"]})
print(result)
# {'subjects': /['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}






