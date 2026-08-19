from langgraph.graph import END, StateGraph
from app.graph.state import GraphState

graph = StateGraph(GraphState)


graph_compiled = graph.compile()
