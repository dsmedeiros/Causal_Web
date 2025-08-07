import Causal_Web.engine.scheduler as scheduler


def test_run_multi_layer_order():
    order = []

    def q_tick():
        order.append("q")

    def c_tick():
        order.append("c")

    def flush():
        order.append("f")

    scheduler.run_multi_layer(q_tick, c_tick, micro_ticks=2, macro_ticks=2, flush=flush)
    assert order == ["q", "q", "f", "c", "q", "q", "f", "c"]
