from pathlib import Path
from app.graph.graph import graph_compiled


def export(path: str = "docs/graph.png") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(graph_compiled.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    export()
    print("wrote docs/graph.png")
