import numpy as np

from q.coin_walk import build_split_step, coin_operator, msd_with_theta_noise


def test_coin_operator_unitary():
    theta = 0.3
    C = coin_operator(theta)
    eye = C.conj().T @ C
    assert np.allclose(eye, np.eye(2))


def test_split_step_builder():
    step = build_split_step(0.1, 0.2)

    def shift_x(state: np.ndarray) -> np.ndarray:
        return state

    def shift_y(state: np.ndarray) -> np.ndarray:
        return state

    state = np.array([1.0, 0.0], dtype=np.complex128)
    out = step(state, shift_x, shift_y)
    assert out.shape == (2,)


def test_theta_noise_ablation():
    ballistic = msd_with_theta_noise((0.35, 0.2), steps=20, noise=0.0)
    diffusive = msd_with_theta_noise((0.35, 0.2), steps=20, noise=0.5)
    assert ballistic > diffusive
