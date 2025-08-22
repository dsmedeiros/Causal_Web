import textwrap

import experiments.bench_optim as bo


def test_optimizer_order(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        textwrap.dedent(
            """
            task,conditional,winner
            t1,False,GA
            t2,False,CMA-ES
            t3,True,MCTS-H

            optimizer,conditional,evals_per_s
            GA,conditional,50
            CMA-ES,conditional,40
            MCTS-H,conditional,60
            GA,unconditional,100
            CMA-ES,unconditional,120
            MCTS-H,unconditional,90
            """
        ).lstrip()
    )

    assert bo.optimizer_order(True, summary)[0] == "MCTS-H"
    assert bo.optimizer_order(False, summary)[0] == "CMA-ES"
