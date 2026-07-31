from pathlib import Path
from statistics import median
from time import perf_counter

from matters.web import graph_payload


KENETT_GRAPH = (
    Path(__file__).parents[1]
    / "examples"
    / "creativity_matters_graph"
    / "creativity_graph.json"
)


def test_creativity_matters_graph_payload_median_is_under_500_ms():
    graph_payload(KENETT_GRAPH)

    timings = []
    payload = None
    for _ in range(5):
        started = perf_counter()
        payload = graph_payload(KENETT_GRAPH)
        timings.append(perf_counter() - started)

    assert median(timings) < 0.5
    assert len(payload["nodes"]) == 1057
    assert len(payload["edges"]) == 1917
