import numpy as np
from dataclasses import dataclass

HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


@dataclass
class MatrixProductState:
    """Simple Matrix Product State for qubit chains.

    Parameters
    ----------
    num_sites:
        Number of sites in the chain.
    chi_max:
        Maximum bond dimension. Singular values beyond this limit are
        truncated during SVD-based compression.
    """

    num_sites: int
    chi_max: int = 16

    def __post_init__(self) -> None:
        self.tensors = [
            np.zeros((1, 2, 1), dtype=np.complex128) for _ in range(self.num_sites)
        ]
        for t in self.tensors:
            t[0, 0, 0] = 1.0

    # ------------------------------------------------------------------
    def apply_unitary(self, unitary: np.ndarray, site: int) -> None:
        """Apply a single-qubit unitary to the given site."""

        tensor = self.tensors[site]
        contracted = np.tensordot(unitary, tensor, axes=[1, 1])  # (2, l, r)
        tensor = np.transpose(contracted, (1, 0, 2))
        self.tensors[site] = tensor
        self._svd_truncate(site)

    # ------------------------------------------------------------------
    def _svd_truncate(self, site: int) -> None:
        if site >= self.num_sites - 1:
            return
        left = self.tensors[site]
        l, p, r = left.shape
        mat = left.reshape(l * p, r)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        chi = min(len(s), self.chi_max)
        u = u[:, :chi]
        s = s[:chi]
        vh = vh[:chi, :]
        norm = np.linalg.norm(s)
        if norm:
            s = s / norm
        self.tensors[site] = u.reshape(l, p, chi)
        right = np.tensordot(np.diag(s) @ vh, self.tensors[site + 1], axes=[1, 0])
        self.tensors[site + 1] = right

    # ------------------------------------------------------------------
    def amplitude(self, bits: list[int]) -> complex:
        """Return amplitude for the computational basis state ``bits``."""

        vec = self.tensors[0][:, bits[0], :]
        for i in range(1, self.num_sites):
            vec = np.tensordot(vec, self.tensors[i][:, bits[i], :], axes=[-1, 0])
        return vec[0]

    # ------------------------------------------------------------------
    def memory_usage(self) -> int:
        """Compute total memory usage in bytes."""

        return int(sum(t.nbytes for t in self.tensors))


def propagate_chain(
    unitaries: list[np.ndarray], psi: np.ndarray, chi_max: int = 16
) -> tuple[np.ndarray, int]:
    """Propagate ``psi`` through a linear chain of ``unitaries`` using an MPS.

    Returns the final two-component state vector and the peak memory
    consumed by the MPS in bytes.
    """

    # TODO: Implement CuPy-backed version when Config.backend == "cupy"
    mps = MatrixProductState(len(unitaries) + 1, chi_max=chi_max)
    for i in range(2):
        mps.tensors[0][0, i, 0] = psi[i]
    peak = mps.memory_usage()
    for idx, u in enumerate(unitaries, start=1):
        mps.apply_unitary(u, idx)
        peak = max(peak, mps.memory_usage())
    zero = [0] * (len(unitaries) + 1)
    zero[-1] = 0
    amp0 = mps.amplitude(zero)
    zero[-1] = 1
    amp1 = mps.amplitude(zero)
    return np.array([amp0, amp1], dtype=np.complex128), peak
