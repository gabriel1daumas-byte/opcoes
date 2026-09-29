"""Enriquecimento da grade, scanners de renda e travas, e métricas de estratégias."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .bs import gregas, vol_implicita, preco_bs_vetor, DIAS_ANO


def enriquecer(df, S, du, r, hv=None):
    """Adiciona vol implícita, gregas e métricas de renda a cada opção."""
    T = max(du, 1) / DIAS_ANO
    df = df.copy()
    ivs, gs = [], []
    for _, o in df.iterrows():
        iv = vol_implicita(o["preco"], S, o["strike"], T, r, o["tipo"])
        ivs.append(iv)
        gs.append(gregas(S, o["strike"], T, r, iv, o["tipo"]))
    df["iv"] = ivs
    g = pd.DataFrame(gs, index=df.index)
    df = pd.concat([df, g], axis=1)
    df["dist_pct"] = df["strike"] / S - 1
    df["valor_intr"] = np.where(df["tipo"] == "CALL", np.maximum(S - df["strike"], 0),
                                np.maximum(df["strike"] - S, 0))
    df["valor_extr"] = df["preco"] - df["valor_intr"]
    if hv:
        df["iv_hv"] = df["iv"] / hv
    return df


# ---------------------------------------------------------------- renda
def anualizar(taxa, du):
    return (1 + taxa) ** (DIAS_ANO / max(du, 1)) - 1


def scanner_renda(df, S, du, min_neg=10):
    """Venda coberta (calls) e venda de put com garantia (puts)."""
    x = df[(df["negocios"] >= min_neg) & (df["preco"] > 0) & df["iv"].notna()].copy()

    c = x[x["tipo"] == "CALL"].copy()
    c["estrategia"] = "Venda coberta"
    c["taxa"] = c["valor_extr"].clip(lower=0) / S         # ganho se o preço ficar parado
    c["retorno_exercido"] = (c["strike"] - S + c["preco"]) / S
    c["protecao"] = c["preco"] / S                        # queda que o prêmio absorve
    c["preco_efetivo"] = S - c["preco"]

    p = x[x["tipo"] == "PUT"].copy()
    p["estrategia"] = "Venda de put"
    p["taxa"] = p["valor_extr"].clip(lower=0) / p["strike"]
    p["retorno_exercido"] = np.nan
    p["protecao"] = 1 - (p["strike"] - p["preco"]) / S    # desconto de compra vs. hoje
    p["preco_efetivo"] = p["strike"] - p["preco"]

    out = pd.concat([c, p])
    out["taxa_aa"] = anualizar(out["taxa"], du)
    out["prob_ficar"] = 1 - out["prob_exerc"]
    return out


# ---------------------------------------------------------------- estratégias
def prob_lognormal(S, T, r, sig, grade):
    """Densidade de S_T (neutra ao risco) nos pontos da grade, normalizada."""
    m = np.log(S) + (r - 0.5 * sig ** 2) * T
    s = sig * np.sqrt(T)
    g = np.maximum(grade, 1e-9)
    pdf = norm.pdf((np.log(g) - m) / s) / (g * s)
    w = pdf * np.gradient(grade)
    return w / w.sum()


def payoff_venc(pernas, grade):
    """Resultado no vencimento por unidade. perna: tipo(CALL/PUT/ACAO), qtd(+compra/-venda), strike, premio."""
    tot = np.zeros_like(grade, dtype=float)
    for p in pernas:
        q, K, pr, t = float(p["qtd"]), float(p.get("strike") or 0), float(p["premio"]), p["tipo"].upper()
        if t == "CALL":
            v = np.maximum(grade - K, 0)
        elif t == "PUT":
            v = np.maximum(K - grade, 0)
        else:
            v = grade
        tot += q * (v - pr)
    return tot


def payoff_hoje(pernas, grade, T, r, sig_padrao):
    tot = np.zeros_like(grade, dtype=float)
    for p in pernas:
        q, pr, t = float(p["qtd"]), float(p["premio"]), p["tipo"].upper()
        if t == "ACAO":
            v = grade
        else:
            sig = p.get("iv")
            sig = sig if sig and np.isfinite(sig) and sig > 0 else sig_padrao
            v = preco_bs_vetor(grade, float(p["strike"]), T, r, sig, t)
        tot += q * (v - pr)
    return tot


def metricas(pernas, S, T, r, sig, n=3000):
    ks = [float(p["strike"]) for p in pernas if p["tipo"].upper() != "ACAO" and p.get("strike")]
    grade = np.union1d(np.linspace(S * 0.001, S * 3.0, n), ks)
    pay = payoff_venc(pernas, grade)
    w = prob_lognormal(S, T, r, sig, grade)
    # detecta ganho/perda ilimitados pela inclinação nas pontas
    incl_dir = pay[-1] - pay[-2]
    ganho_max = np.inf if incl_dir > 1e-9 else pay.max()
    perda_max = -np.inf if incl_dir < -1e-9 else pay.min()
    sinais = np.sign(pay)
    idx = np.where(np.diff(sinais) != 0)[0]
    bes = [grade[i] - pay[i] * (grade[i + 1] - grade[i]) / (pay[i + 1] - pay[i]) for i in idx]
    custo = sum(float(p["qtd"]) * float(p["premio"]) for p in pernas if p["tipo"].upper() != "ACAO")
    return dict(ganho_max=ganho_max, perda_max=perda_max, breakevens=bes,
                prob_lucro=float(w[pay > 0].sum()), valor_esperado=float((w * pay).sum()),
                custo_montagem=custo, grade=grade, payoff=pay)


def scanner_travas(df, S, du, r, sig, min_neg=10, faixa=0.2, max_larg=None):
    """Varre travas verticais (débito e crédito) entre strikes do mesmo vencimento."""
    T = max(du, 1) / DIAS_ANO
    x = df[(df["negocios"] >= min_neg) & (df["preco"] > 0) & df["iv"].notna()
           & (df["strike"].between(S * (1 - faixa), S * (1 + faixa)))]
    grade = np.linspace(S * 0.2, S * 3.0, 3000)
    w = prob_lognormal(S, T, r, sig, grade)
    linhas = []
    for tipo, nomes in (("CALL", ("Trava de alta (call, débito)", "Trava de baixa (call, crédito)")),
                        ("PUT", ("Trava de baixa (put, débito)", "Trava de alta (put, crédito)"))):
        o = x[x["tipo"] == tipo].sort_values("strike").reset_index(drop=True)
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                a, b = o.loc[i], o.loc[j]           # a: strike menor, b: strike maior
                larg = b["strike"] - a["strike"]
                if max_larg and larg > max_larg:
                    break
                if tipo == "CALL":
                    deb = [dict(tipo="CALL", qtd=1, strike=a["strike"], premio=a["preco"]),
                           dict(tipo="CALL", qtd=-1, strike=b["strike"], premio=b["preco"])]
                else:
                    deb = [dict(tipo="PUT", qtd=1, strike=b["strike"], premio=b["preco"]),
                           dict(tipo="PUT", qtd=-1, strike=a["strike"], premio=a["preco"])]
                cred = [dict(p, qtd=-p["qtd"]) for p in deb]
                for nome, pernas in ((nomes[0], deb), (nomes[1], cred)):
                    pay = payoff_venc(pernas, grade)
                    gmax, pmax = pay.max(), pay.min()
                    if gmax <= 0 or pmax >= 0:
                        continue  # preços inconsistentes (arbitragem aparente / dado velho)
                    fluxo = -sum(p["qtd"] * p["premio"] for p in pernas)
                    linhas.append(dict(
                        estrategia=nome, perna1=f"{'C' if pernas[0]['qtd'] > 0 else 'V'} {a['codigo'] if pernas[0]['strike'] == a['strike'] else b['codigo']}",
                        perna2=f"{'C' if pernas[1]['qtd'] > 0 else 'V'} {a['codigo'] if pernas[1]['strike'] == a['strike'] else b['codigo']}",
                        k_menor=a["strike"], k_maior=b["strike"], largura=larg,
                        fluxo_inicial=fluxo, ganho_max=gmax, perda_max=pmax,
                        ganho_risco=gmax / -pmax, prob_lucro=float(w[pay > 0].sum()),
                        valor_esperado=float((w * pay).sum()),
                        neg_min=min(a["negocios"], b["negocios"])))
    out = pd.DataFrame(linhas)
    if not out.empty:
        out["ve_por_risco"] = out["valor_esperado"] / -out["perda_max"]
        # relação ganho/risco absurda quase sempre é preço defasado entre as pernas
        out["alerta"] = np.where((out["ganho_risco"] > 8) | (out["ganho_risco"] < 1 / 8),
                                 "verificar book", "")
    return out
