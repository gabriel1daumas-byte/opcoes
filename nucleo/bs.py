"""Black-Scholes, gregas, volatilidade implícita e calendário B3 (dias úteis)."""
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DIAS_ANO = 252


# ---------------------------------------------------------------- calendário
def _pascoa(ano: int) -> date:
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


@lru_cache(maxsize=None)
def feriados_b3(ano: int) -> tuple:
    p = _pascoa(ano)
    fer = [
        date(ano, 1, 1), p - timedelta(days=48), p - timedelta(days=47),  # carnaval
        p - timedelta(days=2), date(ano, 4, 21), date(ano, 5, 1),
        p + timedelta(days=60), date(ano, 9, 7), date(ano, 10, 12),
        date(ano, 11, 2), date(ano, 11, 15), date(ano, 11, 20),
        date(ano, 12, 24), date(ano, 12, 25), date(ano, 12, 31),
    ]
    return tuple(np.datetime64(d) for d in fer)


def dias_uteis(inicio: date, fim: date) -> int:
    """Dias úteis B3 entre hoje (exclusive) e o vencimento (inclusive)."""
    anos = range(inicio.year, fim.year + 1)
    hol = [h for a in anos for h in feriados_b3(a)]
    return int(np.busday_count(inicio + timedelta(days=1), fim + timedelta(days=1), holidays=hol))


# ---------------------------------------------------------------- Black-Scholes
def _d1d2(S, K, T, r, sig, q=0.0):
    st = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / st
    return d1, d1 - st


def preco_bs(S, K, T, r, sig, tipo, q=0.0):
    """Preço teórico europeu. r e q contínuos (a.a.), T em anos."""
    call = tipo.upper() == "CALL"
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sig, q)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def preco_bs_vetor(S, K, T, r, sig, tipo):
    """Versão vetorizada em S (array) para curvas de payoff."""
    S = np.asarray(S, dtype=float)
    call = tipo.upper() == "CALL"
    if T <= 0 or sig <= 0:
        return np.maximum(S - K, 0) if call else np.maximum(K - S, 0)
    Ss = np.maximum(S, 1e-9)
    d1, d2 = _d1d2(Ss, K, T, r, sig)
    if call:
        return Ss * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - Ss * norm.cdf(-d1)


def gregas(S, K, T, r, sig, tipo, q=0.0):
    """Delta, gama, theta (R$/dia útil), vega (R$ por 1 p.p. de vol), prob. exercício."""
    nan = dict(delta=np.nan, gama=np.nan, theta=np.nan, vega=np.nan, prob_exerc=np.nan)
    if T <= 0 or sig is None or not np.isfinite(sig) or sig <= 0:
        return nan
    call = tipo.upper() == "CALL"
    d1, d2 = _d1d2(S, K, T, r, sig, q)
    pdf = norm.pdf(d1)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    gama = eq * pdf / (S * sig * np.sqrt(T))
    vega = S * eq * pdf * np.sqrt(T) / 100
    if call:
        delta = eq * norm.cdf(d1)
        theta = (-S * eq * pdf * sig / (2 * np.sqrt(T)) - r * K * er * norm.cdf(d2)
                 + q * S * eq * norm.cdf(d1))
        prob = norm.cdf(d2)
    else:
        delta = -eq * norm.cdf(-d1)
        theta = (-S * eq * pdf * sig / (2 * np.sqrt(T)) + r * K * er * norm.cdf(-d2)
                 - q * S * eq * norm.cdf(-d1))
        prob = norm.cdf(-d2)
    return dict(delta=delta, gama=gama, theta=theta / DIAS_ANO, vega=vega, prob_exerc=prob)


def vol_implicita(preco, S, K, T, r, tipo, q=0.0):
    """Vol implícita via Brent. Retorna NaN se o preço estiver fora dos limites."""
    if preco is None or not np.isfinite(preco) or preco <= 0 or T <= 0:
        return np.nan
    call = tipo.upper() == "CALL"
    intr = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if call else \
        max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    teto = S if call else K
    if preco <= intr + 1e-6 or preco >= teto:
        return np.nan
    f = lambda s: preco_bs(S, K, T, r, s, tipo, q) - preco
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=200)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- volatilidade histórica
def vol_historica(fech, janela=21):
    """Vol anualizada (rolling) a partir de uma série de fechamentos."""
    ret = np.log(fech / fech.shift(1))
    return ret.rolling(janela).std() * np.sqrt(DIAS_ANO)
