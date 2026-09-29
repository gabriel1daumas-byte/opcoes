"""Coleta de dados gratuitos: opcoes.net.br (grade de opções), Yahoo Finance
(histórico do ativo) e Banco Central (Selic). Inclui modo demonstração e CSV."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

from .bs import preco_bs, dias_uteis, DIAS_ANO

URL_OPCOES = "https://opcoes.net.br/listaopcoes/completa"
URL_SELIC = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
HEADERS = {"User-Agent": "Mozilla/5.0 (analisador-opcoes)"}

COLUNAS = ["codigo", "tipo", "modelo", "moneyness", "strike", "dist_strike",
           "premio_pct", "preco", "negocios", "volume", "data_cotacao", "vencimento"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- opcoes.net.br
def _get_opcoes(ticker, vencimento=None, listar_venc=True):
    params = {"idAcao": ticker.upper(), "listarVencimentos": str(listar_venc).lower(),
              "cotacoes": "true"}
    if vencimento:
        params["vencimentos"] = vencimento
    r = requests.get(URL_OPCOES, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    j = r.json()
    if not j.get("success", True) or "data" not in j:
        raise RuntimeError("Resposta inesperada do opcoes.net.br")
    return j["data"]


def listar_vencimentos(ticker):
    """Lista de dicts {data, du, mensal} dos vencimentos disponíveis."""
    d = _get_opcoes(ticker, listar_venc=True)
    out = []
    for v in d.get("vencimentos") or []:
        if isinstance(v, dict):
            attrs = v.get("dataAttributes") or {}
            out.append(dict(data=v.get("value"), du=int(_num(attrs.get("du")) or 0),
                            mensal=str(attrs.get("m")) == "1", selecionado=v.get("selected", False)))
        else:
            out.append(dict(data=str(v), du=None, mensal=False, selecionado=False))
    return out


def grade_opcoes(ticker, vencimento):
    """DataFrame com a grade de opções de um vencimento."""
    d = _get_opcoes(ticker, vencimento, listar_venc=False)
    linhas = []
    for i in d.get("cotacoesOpcoes") or []:
        i = list(i) + [None] * (12 - len(i))
        linhas.append([str(i[0]).split("_")[0], str(i[2]).upper(), i[3], i[4], _num(i[5]),
                       _num(i[6]), _num(i[7]), _num(i[8]), _num(i[9]), _num(i[10]), i[11],
                       vencimento])
    df = pd.DataFrame(linhas, columns=COLUNAS)
    df["negocios"] = df["negocios"].fillna(0)
    # negócios de pregões antigos não indicam liquidez hoje: zera para o filtro descartar
    ult = df["data_cotacao"].dropna().max()
    if isinstance(ult, str):
        df.loc[df["data_cotacao"] != ult, "negocios"] = 0
    return df.sort_values(["tipo", "strike"]).reset_index(drop=True)


def spot_da_grade(df):
    """Estima o preço do ativo a partir da 'distância % do strike' ((K-S)/S)."""
    x = df.dropna(subset=["strike", "dist_strike"])
    x = x[np.abs(x["dist_strike"]) < 0.5]
    if x.empty:
        return np.nan
    return float(np.median(x["strike"] / (1 + x["dist_strike"])))


# ---------------------------------------------------------------- Yahoo / BCB
def historico(ticker, periodo="2y"):
    import yfinance as yf
    t = ticker.upper()
    if not t.endswith(".SA"):
        t += ".SA"
    h = yf.Ticker(t).history(period=periodo, auto_adjust=True)
    if h is None or h.empty:
        raise RuntimeError(f"Sem histórico para {t}")
    h.index = pd.to_datetime(h.index).tz_localize(None)
    return h[["Open", "High", "Low", "Close", "Volume"]]


def selic_atual():
    """Meta Selic (% a.a.) do Banco Central."""
    r = requests.get(URL_SELIC, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return float(str(r.json()[-1]["valor"]).replace(",", "."))


# ---------------------------------------------------------------- CSV do usuário
MAPA_CSV = {
    "codigo": ["codigo", "código", "ticker", "ativo", "serie", "série"],
    "tipo": ["tipo", "type"],
    "strike": ["strike", "exercicio", "exercício", "preco_exercicio"],
    "preco": ["preco", "preço", "ultimo", "último", "premio", "prêmio", "last"],
    "negocios": ["negocios", "negócios", "trades", "num_negocios"],
    "vencimento": ["vencimento", "venc", "expiry", "expiration"],
}


def ler_csv(arquivo):
    df = pd.read_csv(arquivo, sep=None, engine="python", decimal=",")
    cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    for alvo, nomes in MAPA_CSV.items():
        achou = next((cols[n] for n in nomes if n in cols), None)
        if achou is not None:
            out[alvo] = df[achou]
    faltam = {"tipo", "strike", "preco", "vencimento"} - set(out.columns)
    if faltam:
        raise ValueError(f"CSV sem as colunas: {', '.join(sorted(faltam))}")
    out["tipo"] = out["tipo"].astype(str).str.upper().str.strip()
    out["tipo"] = out["tipo"].replace({"C": "CALL", "COMPRA": "CALL", "P": "PUT", "VENDA": "PUT"})
    for c in ["strike", "preco", "negocios"]:
        if c in out:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "."), errors="coerce")
    if "negocios" not in out:
        out["negocios"] = 1
    if "codigo" not in out:
        out["codigo"] = out["tipo"].str[0] + out["strike"].round(2).astype(str)
    out["vencimento"] = pd.to_datetime(out["vencimento"], dayfirst=True).dt.strftime("%Y-%m-%d")
    for c in COLUNAS:
        if c not in out:
            out[c] = np.nan
    return out[COLUNAS]


# ---------------------------------------------------------------- modo demonstração
def historico_demo(S0=38.0, vol=0.32, dias=504, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, vol / np.sqrt(DIAS_ANO), dias)
    # regime de vol mais alta no meio da série, para o gráfico ter vida
    ret[200:260] *= 1.8
    fech = S0 * np.exp(np.cumsum(ret) - ret.sum())
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=dias)
    return pd.DataFrame({"Open": fech, "High": fech * 1.01, "Low": fech * 0.99,
                         "Close": fech, "Volume": 1e7}, index=idx)


def vencimentos_demo():
    hoje = date.today()
    out = []
    for m in range(1, 4):
        ano, mes = hoje.year + (hoje.month + m - 1) // 12, (hoje.month + m - 1) % 12 + 1
        # terceira sexta-feira
        d = date(ano, mes, 15)
        while d.weekday() != 4:
            d = d.replace(day=d.day + 1)
        out.append(dict(data=d.isoformat(), du=dias_uteis(hoje, d), mensal=True, selecionado=m == 1))
    return out


def grade_demo(S, vencimento, r=0.14, vol_atm=0.30, seed=1):
    """Grade sintética com sorriso de volatilidade e ruído — só para testar o app."""
    rng = np.random.default_rng(seed)
    du = dias_uteis(date.today(), datetime.strptime(vencimento, "%Y-%m-%d").date())
    T = max(du, 1) / DIAS_ANO
    passo = 0.5 if S < 60 else 1.0
    strikes = np.arange(np.floor(S * 0.75 / passo) * passo, S * 1.25, passo)
    linhas = []
    for tipo, letra in (("CALL", "J"), ("PUT", "V")):
        for K in strikes:
            m = np.log(K / S)
            sig = vol_atm + 0.35 * m ** 2 - 0.08 * m  # sorriso + skew
            teo = preco_bs(S, K, T, r, sig, tipo)
            intr = max(S - K, 0) if tipo == "CALL" else max(K - S, 0)
            extr = max(teo - intr, 0) * (1 + rng.normal(0, 0.04))  # ruído só no extrínseco
            p = round(max(intr + extr, 0.01), 2)
            neg = int(max(0, rng.normal(300 * np.exp(-abs(m) * 12), 30)))
            linhas.append([f"DEMO{letra}{int(K * 10)}", tipo, "E", "", K, (K - S) / S, p / S,
                           p, neg, neg * p * 100, date.today().isoformat(), vencimento])
    return pd.DataFrame(linhas, columns=COLUNAS)
