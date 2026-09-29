"""Monitor intradiário: varre vários ativos, resume oportunidades e compara leituras."""
from datetime import datetime

import numpy as np
import pandas as pd

from . import dados
from .analise import enriquecer, scanner_renda, scanner_travas, scanner_multiplicador
from .bs import vol_historica

# ativos com opções mais líquidas na B3 (edite à vontade no app)
ATIVOS_PADRAO = [
    "PETR4", "VALE3", "BOVA11", "ITUB4", "BBAS3", "BBDC4", "B3SA3", "ABEV3", "PRIO3", "SUZB3",
    "WEGE3", "MGLU3", "GGBR4", "CSNA3", "USIM5", "ITSA4", "BBSE3", "ELET3", "RENT3", "LREN3",
    "HAPV3", "CMIG4", "EMBR3", "VIVT3", "CYRE3", "COGN3", "RADL3", "EQTL3", "PETR3", "SMAL11",
]


def _hv_lote(tickers):
    """HV 21d de vários ativos com um único download do Yahoo."""
    try:
        import yfinance as yf
        h = yf.download([t + ".SA" for t in tickers], period="6mo", auto_adjust=True,
                        progress=False, group_by="ticker", threads=True)
        out = {}
        for t in tickers:
            try:
                c = h[t + ".SA"]["Close"].dropna()
                s = vol_historica(c, 21).dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
            except Exception:
                pass
        return out
    except Exception:
        return {}


def _ler_ativo(t, demo=False, r=0.14):
    """Grade do vencimento padrão (mensal mais próximo) de um ativo — 1 requisição."""
    if demo:
        rng = np.random.default_rng(abs(hash((t, datetime.now().minute))) % 2**32)
        S = float(rng.uniform(10, 120))
        v = dados.vencimentos_demo()[0]
        g = dados.grade_demo(S, v["data"], r=r, vol_atm=float(rng.uniform(0.2, 0.5)),
                             seed=int(rng.integers(1e6)))
        return S, v, g
    _, v, g = dados.vencimentos_e_grade(t)
    return dados.spot_da_grade(g), v, g


def atualizar_cache(tickers, cache, max_req=5, demo=False, r=0.14):
    """Atualiza até `max_req` ativos, começando pelos de dado mais antigo (rodízio).
    cache: dict ativo -> (datetime, S, venc, grade). Retorna dict de erros."""
    erros = {}
    ordem = sorted(tickers, key=lambda t: cache[t][0] if t in cache else datetime.min)
    feitos = 0
    for t in ordem:
        if feitos >= (len(tickers) if demo else max_req):
            break
        if not demo and dados.espera_restante() > 0:
            erros["_limite"] = f"site pediu pausa de {dados.espera_restante():.0f}s; usando os dados já lidos"
            break
        try:
            S, v, g = _ler_ativo(t, demo, r)
            cache[t] = (datetime.now(), S, v, g)
        except dados.LimiteRequisicoes as e:
            erros["_limite"] = str(e) + "; usando os dados já lidos"
            break
        except Exception as e:
            erros[t] = str(e)[:120]
        feitos += 1
    return erros


def analisar(cache, tickers, r, hv, min_neg=10, capital=697.0, meta=5000.0, demo=False):
    """Resumo por ativo + candidatos de alto risco de todos os ativos em cache."""
    resumo, ops, riscos = [], [], []
    mult = meta / capital
    agora = datetime.now()
    for t in tickers:
        if t not in cache:
            continue
        ts, S, v, g = cache[t]
        if not np.isfinite(S) or g.empty:
            continue
        du = max(int(v["du"] or 1), 1)
        g = g[(g["negocios"] >= min_neg) & (g["preco"] > 0)]
        if g.empty:
            continue
        e = enriquecer(g, S, du, r, hv.get(t))
        liq = e[e["iv"].notna()]
        if liq.empty:
            continue
        atm = liq.assign(d=(liq["strike"] - S).abs()).sort_values("d").head(4)
        iv_atm = float(atm["iv"].median())
        h = hv.get(t, np.nan)
        if demo:
            h = iv_atm * float(np.random.default_rng(len(t)).uniform(0.7, 1.4))
        sig = h if np.isfinite(h) else iv_atm

        ren = scanner_renda(e, S, du, min_neg)
        ren = ren[ren["delta"].abs().between(0.2, 0.4)]
        top_ren = ren.sort_values("taxa_aa", ascending=False).head(1)
        tr = scanner_travas(e, S, du, r, sig, min_neg, 0.15)
        if not tr.empty:
            tr = tr[tr["alerta"] == ""]
        top_tr = tr.sort_values("ve_por_risco", ascending=False).head(1) if not tr.empty else tr
        ar = scanner_multiplicador(e, S, du, r, sig, capital, mult, min_neg)
        ar = ar[ar["qtd"] > 0] if not ar.empty else ar
        if not ar.empty:
            ar = ar.sort_values("prob_hoje", ascending=False)
            riscos.append(ar.head(5).assign(ativo=t, preco_ativo=S, vencimento=v["data"], du=du,
                                            vol_usada=sig, lido_as=ts))
        top_ar = ar.head(1)

        razao = iv_atm / h if np.isfinite(h) else np.nan
        sinal = ("VENDER vol (prêmios caros)" if razao > 1.2 else
                 "COMPRAR vol (prêmios baratos)" if razao < 0.8 else "neutro") if np.isfinite(razao) else "—"
        linha = dict(ativo=t, lido_as=ts.strftime("%H:%M"), idade_min=int((agora - ts).total_seconds() // 60),
                     preco=S, vencimento=v["data"], du=du, negocios=int(liq["negocios"].sum()),
                     iv_atm=iv_atm, hv21=h, iv_hv=razao, sinal=sinal,
                     renda_melhor=f"{top_ren.iloc[0]['estrategia']} {top_ren.iloc[0]['codigo']}" if len(top_ren) else "",
                     renda_taxa_aa=float(top_ren.iloc[0]["taxa_aa"]) if len(top_ren) else np.nan,
                     trava_melhor=f"{top_tr.iloc[0]['estrategia']} {top_tr.iloc[0]['perna1']} / {top_tr.iloc[0]['perna2']}" if len(top_tr) else "",
                     trava_ve_risco=float(top_tr.iloc[0]["ve_por_risco"]) if len(top_tr) else np.nan,
                     trava_prob=float(top_tr.iloc[0]["prob_lucro"]) if len(top_tr) else np.nan,
                     risco_melhor=top_ar.iloc[0]["codigo"] if len(top_ar) else "",
                     risco_prob_hoje=float(top_ar.iloc[0]["prob_hoje"]) if len(top_ar) else np.nan)
        liq_n = np.log10(max(linha["negocios"], 1)) / 4
        vol_n = min(abs(np.log(razao)) / np.log(1.5), 1) if np.isfinite(razao) else 0
        tr_n = min(max(linha["trava_ve_risco"], 0) / 0.3, 1) if np.isfinite(linha["trava_ve_risco"]) else 0
        linha["nota"] = round(100 * (0.35 * min(liq_n, 1) + 0.35 * vol_n + 0.30 * tr_n))
        resumo.append(linha)
        for _, o in liq.iterrows():
            ops.append(dict(ativo=t, codigo=o["codigo"], tipo=o["tipo"], strike=o["strike"],
                            preco=o["preco"], negocios=o["negocios"], iv=o["iv"]))

    res = pd.DataFrame(resumo)
    if not res.empty:
        res = res.sort_values("nota", ascending=False).reset_index(drop=True)
    rk = pd.concat(riscos, ignore_index=True) if riscos else pd.DataFrame()
    if not rk.empty:
        rk = rk.sort_values("prob_hoje", ascending=False).reset_index(drop=True)
    return res, pd.DataFrame(ops), rk


def hv_lote(tickers):
    return _hv_lote(tickers)


def comparar(atual, anterior, ops_atual, ops_anterior):
    """Gera alertas comparando duas leituras."""
    alertas = []
    if anterior is None or anterior.empty or atual.empty:
        return alertas
    a = atual.set_index("ativo")
    b = anterior.set_index("ativo")
    for t in a.index.intersection(b.index):
        x, y = a.loc[t], b.loc[t]
        dpx = x["preco"] / y["preco"] - 1
        div = x["iv_atm"] - y["iv_atm"]
        if abs(dpx) >= 0.01:
            alertas.append((t, f"preço {'subiu' if dpx > 0 else 'caiu'} {dpx:+.1%} desde a última leitura"))
        if abs(div) >= 0.02:
            alertas.append((t, f"IV ATM {'subiu' if div > 0 else 'caiu'} {div * 100:+.1f} p.p. "
                               f"({y['iv_atm']:.0%} → {x['iv_atm']:.0%})"))
        if x["sinal"] != y["sinal"]:
            alertas.append((t, f"sinal de vol mudou: {y['sinal']} → {x['sinal']}"))
        if x["nota"] - y["nota"] >= 15:
            alertas.append((t, f"nota de oportunidade subiu {y['nota']:.0f} → {x['nota']:.0f}"))
    # volume anormal por série
    if ops_anterior is not None and not ops_anterior.empty and not ops_atual.empty:
        m = ops_atual.merge(ops_anterior[["codigo", "negocios", "preco"]], on="codigo", how="left",
                            suffixes=("", "_ant"))
        m["neg_ant"] = m["negocios_ant"].fillna(0)
        m["salto"] = m["negocios"] - m["neg_ant"]
        quentes = m[(m["salto"] >= 200) & (m["negocios"] >= 2 * m["neg_ant"].clip(lower=1))]
        for _, o in quentes.sort_values("salto", ascending=False).head(8).iterrows():
            dp = o["preco"] / o["preco_ant"] - 1 if o["preco_ant"] and np.isfinite(o["preco_ant"]) else np.nan
            txt = f"{o['codigo']}: +{o['salto']:.0f} negócios desde a última leitura"
            if np.isfinite(dp):
                txt += f", prêmio {dp:+.0%}"
            alertas.append((o["ativo"], txt))
    return alertas


def comparar_risco(atual, anterior):
    """Alertas da aba Alto risco: chance de bater a meta subindo, séries novas no top 10."""
    al = []
    if anterior is None or anterior.empty or atual is None or atual.empty:
        return al
    a = atual.head(10).set_index("codigo")
    b = anterior.set_index("codigo")
    for c, x in a.iterrows():
        if c not in b.index:
            al.append((x["ativo"], f"{c} entrou no top 10 (chance hoje {x['prob_hoje']:.2%}, prêmio R$ {x['preco']:.2f})"))
            continue
        y = b.loc[c]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[0]
        dp = x["prob_hoje"] - y["prob_hoje"]
        if dp >= 0.005:
            al.append((x["ativo"], f"{c}: chance de bater a meta hoje subiu {y['prob_hoje']:.2%} → {x['prob_hoje']:.2%}"))
        if y["preco"] > 0 and abs(x["preco"] / y["preco"] - 1) >= 0.25:
            al.append((x["ativo"], f"{c}: prêmio {x['preco'] / y['preco'] - 1:+.0%} (R$ {y['preco']:.2f} → R$ {x['preco']:.2f})"))
    return al
