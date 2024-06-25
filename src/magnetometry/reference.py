#!/usr/bin/env python3

import collections
from itertools import cycle

GroundState = collections.namedtuple("GroundState", ["S", "L", "J", "g"])


def hund(shell, n):
    shell = {"s": 0, "p": 1, "d": 2, "f": 3}.get(shell, shell)
    filled = collections.defaultdict(int)
    levels = cycle(reversed(range(-shell, shell + 1)))
    for i, _ in zip(levels, range(n)):
        filled[i] += 1
    S = sum(v % 2 for k, v in filled.items()) / 2
    L = sum(k * v for k, v in filled.items())
    J = abs(L - S) if n < shell * 2 + 1 else L + S
    g = 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1)) if J else 0
    return GroundState(S, L, J, g)


def effective_moment(ground_state):
    # free ion
    return ground_state.g * (ground_state.J * (ground_state.J + 1)) ** 0.5


rare_earths = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]

ground_states = {elem: hund("f", i) for i, elem in enumerate(rare_earths)}

lattice_constants = {  # angstrom
    "LaN": 5.305,
    "CeN": 5.022,
    "PrN": 5.135,
    "NdN": 5.132,
    "PmN": 5.070,  # guess
    "SmN": 5.035,
    "EuN": 5.017,
    "GdN": 4.974,
    "TbN": 4.920,
    "DyN": 4.895,
    "HoN": 4.865,
    "ErN": 4.842,
    "TmN": 4.800,
    "YbN": 4.781,
    "LuN": 4.760,
}

n_densities = {k: 4 / (v * 1e-8) ** 3 for k, v in lattice_constants.items()}

# grams/cm^3, emu/Oe/cm^3
Substrate = collections.namedtuple("Substrate", ["density", "susceptibility"])
substrates = {
    "Si": Substrate(2.328, -2.968e-7),
    "FS": Substrate(2.2, -8.426e-7),
}
