"""
tvl.py -- reflection, transmission and absorption spectra of a thin vapor layer
between two dielectric windows, with quenching of the atomic polarization at the
walls.

Model: A. V. Ermolaev and T. A. Vartanyan, Phys. Rev. A 105, 013518 (2022).
The even and odd eigenmodes of the layer [their Eqs. (27)-(28)] are computed
exactly, i.e. to all orders in the optical density m, and the reflection and
transmission amplitudes follow from the boundary conditions [their Eqs. (31)-(33)].

Parameters (dimensionless; frequencies in units of the Doppler width k v_T):
    l_lam   layer thickness l / lambda
    Omega   detuning (omega - omega_0) / (k v_T); Omega > 0: laser above resonance
    Gamma   homogeneous half-width gamma / (k v_T)
    m       optical density 2 sqrt(pi) N d^2 / (hbar k v_T)  (Gaussian units)
    n1, n2  refractive indices of the front window (light comes from it) and of
            the rear window; real, i.e. lossless
    N       number of grid intervals across the layer (even); default grid_size()

Outputs: R = |r|^2, T = (n2/n1) |t|^2, A = 1 - R - T (absorbed
in the vapor).

Typical use (see example.ipynb if you want to directly run the simulations):
    R, T, A = spectra(l_lam=5/3, Omega=Om, Gamma=0.01, m=0.002, n1=1.5, n2=1.5)
    fig, axes = plot_spectra(Om, R, T, A, l_lam=5/3, Gamma=0.01, m=0.002, n1=1.5, n2=1.5)
"""
from fractions import Fraction

import numpy as np

__all__ = ['spectra', 'empty_cell', 'plot_spectra', 'rt_amplitudes', 'boundary_numbers', 'grid_size']

SQPI = np.sqrt(np.pi)


# ============================ 1. special function ==============================
def I0(b):
    """I0(b) = int_0^inf exp(-nu^2 - b/nu) dnu  for complex b with Re b >= 0.

    Trapezoid rule in t = ln(nu) on 4200 points (nu from 1.4e-6 to 6.7) the range 
    is specifically seleceted such that the relative error is about 1e-12."""
    b = np.atleast_1d(b)
    t = np.linspace(-13.5, 1.9, 4200)
    nu = np.exp(t)
    F = np.exp(-nu**2)[None, :] * np.exp(-np.outer(b, 1.0 / nu)) * nu[None, :]
    dt = t[1] - t[0]
    return dt * (F.sum(axis=1) - 0.5 * F[:, 0] - 0.5 * F[:, -1])


def G(a, eta):
    """G(a) = int_0^a Kern(u) du, the exact antiderivative of the
    velocity-averaged kernel Kern(u) = -2i int_0^inf exp(-nu^2 - eta u/nu) dnu/nu:

        G(a) = -(2i/eta) [sqrt(pi)/2 - I0(eta a)],   eta = Gamma - i Omega.

    a is a flight distance (in units of 1/k) and G(a) the polarization that
    atoms starting from zero coherence build up while crossing a distance a of
    a unit field."""
    return -(2j / eta) * (SQPI / 2 - I0(eta * np.asarray(a)))


# ============================ 2. discretization ================================
def grid_size(l_lam):
    """Default number of grid intervals: even number, about 100 per wavelength, at least 60.
    The error of the spectra is O(h^2), about 1e-5 in R on this grid. Actually, we can go lower..."""
    return max(60, 2 * int(np.ceil(50 * l_lam)))


def build_K(eta, N, h):
    """Matrix of the polarization source, g_i = sum_j K[i, j] E_j, where

        g(xi) = int_0^phi Kern(|xi - s|) E(s) ds.

    E is taken constant on the cell of each node (half cells at the walls) and
    the kernel is integrated over each cell exactly, so every entry is G(distance
    to the far cell edge) - G(distance to the near cell edge)."""
    d = np.arange(N + 1)
    Gp = G((d + 0.5) * h, eta)                       # G at (d + 1/2) h
    Gn = G(d * h, eta)                               # G at d h (Gn[0] is never used)
    idx = np.abs(np.subtract.outer(d, d))            # distance |i - j| in cells
    K = Gp[idx] - Gp[np.maximum(idx - 1, 0)]         # interior cells
    np.fill_diagonal(K, 2 * Gp[0])                   # the cell around node i itself
    K[:, 0] = np.where(d == 0, Gp[0], Gn[d] - Gp[np.maximum(d - 1, 0)])          # front half cell
    K[:, N] = np.where(d == N, Gp[0], Gn[N - d] - Gp[np.maximum(N - d - 1, 0)])  # rear half cell
    return K


def build_S_wD(N, h, xi):
    """Matrix S and front-wall weights wD (trapezoid rule):

        (S g)(xi_i) = int_{phi/2}^{xi_i} sin(xi_i - s) g(s) ds,
        E'(0) = E0'(0) + m wD . g  with  wD . g = -int_0^{phi/2} cos(s) g(s) ds.

    S is anchored at the centre of the layer, which fixes the parity of the modes."""
    mid = N // 2
    S = np.zeros((N + 1, N + 1), dtype=complex)
    for i in range(N + 1):
        if i == mid:
            continue                                 # (S g) vanishes at the anchor
        j0, j1, sgn = (mid, i, 1.0) if i > mid else (i, mid, -1.0)
        w = np.full(j1 - j0 + 1, h)
        w[0] = w[-1] = h / 2
        S[i, j0:j1 + 1] = sgn * w * np.sin(xi[i] - xi[j0:j1 + 1])
    w = np.full(mid + 1, h)
    w[0] = w[-1] = h / 2
    wD = np.zeros(N + 1)
    wD[:mid + 1] = -w * np.cos(xi[:mid + 1])
    return S, wD


# ============================ 3. solver ========================================
def boundary_numbers(l_lam, Omega, Gamma, m, N=None):
    """Values and slopes at the front wall of the even and odd eigenmodes,
    (I1, I2, I3, I4) = (Ee(0), Eo(0), Ee'(0), Eo'(0)) [their Eq. (34)], for every
    detuning. The modes solve (1 - m S K) E = E0 with the seeds
    E0 = cos(xi - phi/2) and sin(xi - phi/2), i.e. exactly in m.

    Returns a complex array of shape (len(Omega), 4)."""
    N = grid_size(l_lam) if N is None else int(N)
    if N % 2:
        raise ValueError('N must be even: the centre of the layer must be a grid node')
    phi = 2 * np.pi * l_lam
    xi = np.linspace(0.0, phi, N + 1)
    h = xi[1] - xi[0]
    S, wD = build_S_wD(N, h, xi)                     # independent of the detuning
    seeds = np.column_stack([np.cos(xi - phi / 2), np.sin(xi - phi / 2)]).astype(complex)
    Omega = np.atleast_1d(np.asarray(Omega, dtype=float))
    I = np.empty((Omega.size, 4), dtype=complex)
    for k, Om in enumerate(Omega):
        K = build_K(Gamma - 1j * Om, N, h)           # eta = Gamma - i Omega
        Ee, Eo = np.linalg.solve(np.eye(N + 1) - m * (S @ K), seeds).T
        I[k] = (Ee[0], Eo[0],
                np.sin(phi / 2) + m * (wD @ (K @ Ee)),
                np.cos(phi / 2) + m * (wD @ (K @ Eo)))
    return I


def rt_amplitudes(I, n1, n2):
    """Reflection and transmission amplitudes r, t from the boundary numbers. 
    Note that the the windows only enter in this part, as the computation before was
    done to compute the eigenmodes of the vapor layer."""
    I1, I2, I3, I4 = I[..., 0], I[..., 1], I[..., 2], I[..., 3]
    P, Q, U = I1 * I4 + I2 * I3, I1 * I2, I3 * I4
    den = (n1 + n2) * P + 2j * (n1 * n2 * Q - U)
    r = ((n1 - n2) * P + 2j * (n1 * n2 * Q + U)) / den
    t = 2 * n1 * (I1 * I4 - I2 * I3) / den
    return r, t


def spectra(l_lam, Omega, Gamma, m, n1, n2, N=None):
    """Reflectance R, transmittance T and absorptance A versus detuning.

    Parameters: see the module docstring. Omega may be a scalar or an array.
    Returns three real arrays (fractions of the incident flux, not %)."""
    if np.iscomplexobj(n1) or np.iscomplexobj(n2):
        raise ValueError('n1 and n2 must be real: T and A are defined for lossless windows')
    r, t = rt_amplitudes(boundary_numbers(l_lam, Omega, Gamma, m, N), n1, n2)
    R = np.abs(r) ** 2
    T = (n2 / n1) * np.abs(t) ** 2
    return R, T, 1.0 - R - T


def empty_cell(l_lam, n1, n2):
    """Reflectance R0 and transmittance T0 of the empty cell (m = 0, Fabry-Perot).
    Without atoms the modes are the seeds, so the boundary numbers are those of
    cos(xi - phi/2) and sin(xi - phi/2) at xi = 0. l_lam may be an array."""
    phi = 2 * np.pi * np.asarray(l_lam, dtype=float)
    I = np.stack([np.cos(phi / 2), -np.sin(phi / 2), np.sin(phi / 2), np.cos(phi / 2)], axis=-1)
    r, t = rt_amplitudes(I, n1, n2)
    return np.abs(r) ** 2, (n2 / n1) * np.abs(t) ** 2


# ============================ 4. plot ==========================================
def _thickness_label(l_lam):
    """'5/3' for l_lam = 5/3, otherwise the decimal value."""
    f = Fraction(l_lam).limit_denominator(100)
    if f.denominator > 1 and abs(f.numerator / f.denominator - l_lam) < 1e-9:
        return '{}/{}'.format(f.numerator, f.denominator)
    return '{:g}'.format(l_lam)


def plot_spectra(Omega, R, T, A, l_lam, Gamma, m, n1, n2):
    """R, T and A in % versus Omega/Gamma, in one row, with the parameters in the title.
    Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    x = np.asarray(Omega) / Gamma
    for ax, y, sym, name in zip(axes, (R, T, A), 'RTA', ('Reflection', 'Transmission', 'Absorption')):
        ax.plot(x, 100 * np.asarray(y), color='k', lw=1.6)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r'$(\omega - \omega_0)/\gamma$', fontsize=12)
        ax.set_ylabel(r'${}$ (%)'.format(sym), fontsize=12)
        ax.set_xlim(x.min(), x.max())
        ax.grid(ls=':', lw=0.6, color='0.75')
    fig.suptitle(r'Thin vapor layer, quenching walls:   $l = {}\,\lambda$,   $\Gamma = {:g}$,   '
                 r'$m = {:g}$,   $n_1 = {:g}$,   $n_2 = {:g}$'.format(
                     _thickness_label(l_lam), Gamma, m, n1, n2), fontsize=13)
    return fig, axes
