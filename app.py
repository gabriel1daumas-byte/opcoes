"""Analisador de Opções B3 — rode com:  streamlit run app.py"""
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nucleo import dados, monitor
from nucleo.analise import (enriquecer, scanner_renda, scanner_travas, metricas, payoff_hoje,
                            scanner_multiplicador)
from nucleo.bs import vol_historica, dias_uteis, DIAS_ANO

st.set_page_config(page_title="Analisador de Opções B3", layout="wide")

AZUL, LARANJA, CINZA, VERDE, VERMELHO = "#2a6fdb", "#e8833a", "#8a8f98", "#2e9e5b", "#d64545"
PCT = "{:.1%}".format


# ================================================================ cache de dados
@st.cache_data(ttl=600, show_spinner=False)
def c_venc_grade(ticker):
    return dados.vencimentos_e_grade(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def c_grade(ticker, venc):
    return dados.grade_opcoes(ticker, venc)


@st.cache_data(ttl=3600, show_spinner=False)
def c_hist(ticker):
    return dados.historico(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def c_selic():
    return dados.selic_atual()


# ================================================================ barra lateral
st.sidebar.title("Analisador de Opções B3")
fonte = st.sidebar.radio("Fonte dos dados", ["Online (gratuito)", "Arquivo CSV", "Demonstração"],
                         help="Online: opcoes.net.br + Yahoo Finance + Banco Central.")
ticker = st.sidebar.text_input("Ativo-objeto", "PETR4").upper().strip()

try:
    selic_auto = c_selic()
except Exception:
    selic_auto = 15.0
selic = st.sidebar.number_input("Selic (% a.a.)", 0.0, 50.0, float(selic_auto), 0.25,
                                help="Buscada no Banco Central; ajuste se quiser.")
r = float(np.log(1 + selic / 100))  # taxa contínua
min_neg = st.sidebar.number_input("Mínimo de negócios no dia", 0, 10000, 10, 5,
                                  help="Filtra opções sem liquidez (preço pode estar velho).")
so_opcoes = st.sidebar.toggle("Só opções (sem operar ações)", True,
                              help="Esconde venda coberta, venda de put para comprar a ação e estruturas com ações.")
st.sidebar.caption("Ferramenta de estudo. Não é recomendação de investimento. "
                   "Custos de corretagem, emolumentos e IR não estão incluídos.")



def br(v, casas=2):
    """Número no formato brasileiro: 5.000,00"""
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_tabela(df, pct=(), money=(), num=()):
    if "vencimento" in df:
        df = df.copy()
        df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce").dt.strftime("%d/%m/%Y")
    f = {c: "{:.1%}" for c in pct if c in df}
    f.update({c: "R$ {:,.2f}" for c in money if c in df})
    f.update({c: "{:.3f}" for c in num if c in df})
    if "mult_realista" in df:
        f["mult_realista"] = "{:.1f}x"
    sty = df.style.format(f, na_rep="—")
    cores = {"Realista": "#2e9e5b", "Possível": "#7cb342", "Ousado": "#e8833a",
             "Muito ousado": "#d64545", "Praticamente impossível": "#8a8f98"}
    metas = [c for c in df.columns if c in ("meta_hoje", "meta_venc", "risco_meta")]
    if metas:
        sty = sty.map(lambda v: f"color: {cores.get(v, 'inherit')}; font-weight: 600", subset=metas)
    return sty


# ================================================================ abas principais
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Sao_Paulo")
DEMO = fonte == "Demonstração"


def pregao_aberto(agora):
    return agora.weekday() < 5 and (10, 0) <= (agora.hour, agora.minute) < (17, 0)


if DEMO:
    st.warning("Modo demonstração: dados sintéticos, apenas para conhecer o app.")
top = st.tabs(["Monitor do dia", "Alto risco", "Outros menus"])
OUT = top[2]

ss = st.session_state
if ss.get("fonte_ant") != fonte:            # trocou a fonte: zera o monitor
    for k_ in ("cache", "leituras", "alertas", "alertas_risco", "hv"):
        ss.pop(k_, None)
    ss["fonte_ant"] = fonte
ss.setdefault("cache", {})         # ativo -> (hora, S, venc, grade)
ss.setdefault("leituras", [])      # (hora, resumo, ops, risco)
ss.setdefault("alertas", [])       # (hora, ativo, texto) — monitor
ss.setdefault("alertas_risco", []) # (hora, ativo, texto) — alto risco
ss.setdefault("hv", {})

with top[0]:
    st.subheader("Monitor do dia — oportunidades em vários ativos")
    m1, m2, m3 = st.columns([4, 1, 1])
    ativos_txt = m1.text_area("Ativos (separados por vírgula)", ", ".join(monitor.ATIVOS_PADRAO), height=68)
    intervalo = m2.selectbox("Atualizar a cada", [0, 5, 10, 15, 30], index=2,
                             format_func=lambda x: "manual" if x == 0 else f"{x} min")
    por_rodada = m3.number_input("Ativos por rodada", 1, 30, 5,
                                 help="O site gratuito bloqueia muitas consultas seguidas (HTTP 429). "
                                      "O app atualiza alguns ativos por rodada, em rodízio.")
with top[1]:
    st.subheader("Alto risco — todas as oportunidades de multiplicar o capital")
    ca, cb, cc = st.columns([1, 1, 2])
    capital_g = ca.number_input("Capital (R$)", 10.0, 1e7, 697.0, 10.0, key="capital_g")
    meta_g = cb.number_input("Meta (R$)", 10.0, 1e8, 5000.0, 100.0, key="meta_g")
    cc.markdown(f"<div style='padding-top:2rem'>Precisa multiplicar por <b>{meta_g / capital_g:.1f}x</b></div>",
                unsafe_allow_html=True)
ATIVOS = [a.strip().upper() for a in ativos_txt.replace(";", ",").split(",") if a.strip()]


def garantir_leitura(forcar=False):
    """Faz uma nova rodada se a última estiver velha (compartilhada pelas duas abas)."""
    agora = datetime.now(TZ)
    ult = ss["leituras"][-1][0] if ss["leituras"] else None
    velha = ult is None or (intervalo and (agora - ult).total_seconds() >= intervalo * 60 - 5)
    params = (capital_g, meta_g, min_neg, so_opcoes)
    if not (forcar or velha):
        if ss.get("params") != params and ss["leituras"]:   # mudou capital/meta: só recalcula
            res, ops, rk = monitor.analisar(ss["cache"], ATIVOS, r, ss["hv"], min_neg, capital_g, meta_g, DEMO)
            ss["leituras"][-1] = (ss["leituras"][-1][0], res, ops, rk)
            ss["params"] = params
        return
    ss["params"] = params
    with st.spinner("Atualizando dados..."):
        if not ss["hv"] and not DEMO:
            ss["hv"] = monitor.hv_lote(ATIVOS)
        ss["mon_erros"] = monitor.atualizar_cache(ATIVOS, ss["cache"], por_rodada, DEMO, r)
        res, ops, rk = monitor.analisar(ss["cache"], ATIVOS, r, ss["hv"], min_neg, capital_g, meta_g, DEMO)
    hora = agora.strftime("%H:%M")
    if ss["leituras"]:
        _, res0, ops0, rk0 = ss["leituras"][-1]
        for a_, txt in monitor.comparar(res, res0, ops, ops0):
            ss["alertas"].insert(0, (hora, a_, txt))
        for a_, txt in monitor.comparar_risco(rk, rk0):
            ss["alertas_risco"].insert(0, (hora, a_, txt))
    ss["leituras"].append((agora, res, ops, rk))
    del ss["leituras"][:-60]
    try:
        if not res.empty:
            import os
            res.assign(hora=agora.strftime("%Y-%m-%d %H:%M")).to_csv(
                "historico_monitor.csv", mode="a", index=False, header=not os.path.exists("historico_monitor.csv"))
    except Exception:
        pass


def status_linha():
    agora = datetime.now(TZ)
    ts = ss["leituras"][-1][0]
    n = len(ss["cache"])
    st.caption(f"Última rodada: {ts:%H:%M:%S} · {n}/{len(ATIVOS)} ativos já lidos · "
               f"{'pregão ABERTO' if pregao_aberto(agora) else 'pregão FECHADO (último negócio)'} · "
               f"{'automático a cada ' + str(intervalo) + ' min' if intervalo else 'manual'}")
    errs = ss.get("mon_erros") or {}
    if "_limite" in errs:
        st.warning(errs["_limite"])
    outros = {k: v for k, v in errs.items() if k != "_limite"}
    if outros:
        with st.expander(f"{len(outros)} ativo(s) com erro"):
            st.write(outros)


def tabela_alertas(lista, vazio):
    if lista:
        st.dataframe(pd.DataFrame(lista[:50], columns=["hora", "ativo", "o que mudou"]), hide_index=True,
                     width="stretch", height=min(36 * len(lista[:50]) + 38, 300))
    else:
        st.caption(vazio)


RUN = f"{intervalo}m" if intervalo else None


@st.fragment(run_every=RUN)
def painel_monitor():
    garantir_leitura(st.button("Atualizar agora", type="primary", key="bt_mon"))
    if not ss["leituras"]:
        return
    status_linha()
    _, res, _, _ = ss["leituras"][-1]
    if res.empty:
        st.info("Ainda sem dados suficientes. Aguarde a próxima rodada.")
        return
    k = st.columns(3)
    for i, (_, x) in enumerate(res.head(3).iterrows()):
        k[i].metric(f"#{i + 1} {x['ativo']}", f"nota {x['nota']:.0f}",
                    f"{x['sinal']} · IV/HV {x['iv_hv']:.2f}" if np.isfinite(x["iv_hv"]) else x["sinal"],
                    delta_color="off")
    st.markdown("**Alertas do dia**")
    tabela_alertas(ss["alertas"], "Os alertas aparecem a partir da segunda leitura de cada ativo (preço ±1%, "
                                  "IV ±2 p.p., mudança de sinal de vol, salto de negócios, nota subindo).")
    st.markdown("**Ranking de oportunidades**")
    cols = ["ativo", "vencimento", "nota", "lido_as", "preco", "du", "negocios", "iv_atm", "hv21", "iv_hv", "sinal",
            "renda_melhor", "renda_taxa_aa", "trava_melhor", "trava_ve_risco", "trava_prob",
            "risco_melhor", "risco_meta", "risco_prob_hoje", "risco_valor_realista"]
    if so_opcoes:
        cols = [c_ for c_ in cols if not c_.startswith("renda_")]
    st.dataframe(fmt_tabela(res[cols], pct=["iv_atm", "hv21", "renda_taxa_aa", "trava_prob", "risco_prob_hoje"],
                            money=["preco", "risco_valor_realista"], num=["iv_hv", "trava_ve_risco"]),
                 hide_index=True, width="stretch", height=420)
    st.caption("**nota** 0–100: liquidez + distância entre IV e vol histórica + qualidade da melhor trava "
               "(valor esperado por R\\$ de risco, usando a HV). **sinal**: IV/HV > 1,2 favorece vender prêmio; "
               "< 0,8 favorece comprar. **lido_as**: horário do dado daquele ativo. "
               "**risco_meta**: quão realista é bater a meta hoje na melhor opção de alto risco; "
               "**risco_valor_realista**: quanto o capital viraria nela com um movimento típico do dia. "
               "Para detalhar um ativo, use a barra lateral e a aba Outros menus.")
    hist_iv = [l[1].assign(hora=l[0]) for l in ss["leituras"] if not l[1].empty]
    if len(hist_iv) >= 2:
        st.markdown("**IV ATM ao longo do dia (top 5)**")
        h = pd.concat(hist_iv)
        fig = go.Figure()
        for a_ in res.head(5)["ativo"]:
            x = h[h["ativo"] == a_]
            fig.add_scatter(x=x["hora"], y=x["iv_atm"], name=a_, mode="lines+markers")
        fig.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(t=10, b=30), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")


@st.fragment(run_every=RUN)
def painel_risco():
    st.error("Compra de opções fora do dinheiro: o resultado mais provável é **perder 100% do valor aplicado**. "
             "Use apenas dinheiro que você pode perder sem prejudicar suas contas.")
    garantir_leitura(st.button("Atualizar agora", type="primary", key="bt_risco"))
    if not ss["leituras"]:
        return
    status_linha()
    _, _, _, rk = ss["leituras"][-1]
    mult = meta_g / capital_g
    st.caption(f"Capital R\\$ {br(capital_g)} → meta R\\$ {br(meta_g)} = **{mult:.1f}x**. "
               f"Probabilidades com a vol histórica 21d de cada ativo.")
    if rk.empty:
        st.info("Nenhuma opção líquida cabe no capital informado (lote mínimo de 100).")
        return
    best = rk.iloc[0]
    k = st.columns(3)
    k[0].metric("Melhor chance de bater a meta HOJE", monitor.fmt_chance(best["prob_hoje"]), f"{best['codigo']} ({best['ativo']})",
                delta_color="off")
    k[1].metric("Melhor chance até o vencimento", f"{rk['prob_venc'].max():.2%}")
    k[2].metric("Retorno esperado mediano (modelo)", f"{rk['retorno_esperado'].median():+.0%}")
    if best["prob_hoje"] > 0:
        st.info(f"Na melhor opção de todos os ativos, a chance de bater a meta hoje é ≈ 1 em "
                f"{br(1 / best['prob_hoje'], 0)}. Na grande maioria das vezes, o capital inteiro é perdido.")
    st.markdown("**Alertas de alto risco**")
    tabela_alertas(ss["alertas_risco"], "Aparecem a partir da segunda leitura, só para séries com chance de pelo "
                                        "menos 0,5% hoje: chance subindo, série nova no top 10, prêmio ±25%.")
    if best["prob_hoje"] < monitor.CHANCE_MIN_ALERTA:
        st.warning("Nenhuma opção tem chance relevante (≥ 0,5%) de bater a meta hoje. Com a meta atual, o movimento "
                   "necessário é grande demais para um pregão.")
    st.markdown("**Ranking (todas as séries, todos os ativos)**")
    cols = ["ativo", "codigo", "vencimento", "meta_hoje", "valor_realista", "mult_realista", "tipo", "strike",
            "preco", "negocios", "qtd", "custo", "mov_hoje", "prob_hoje", "meta_venc",
            "mov_venc", "prob_venc", "prob_lucro_venc", "retorno_esperado", "iv", "preco_ativo", "du"]
    st.dataframe(fmt_tabela(rk[cols], pct=["mov_hoje", "prob_hoje", "mov_venc", "prob_venc", "prob_lucro_venc",
                                            "retorno_esperado", "iv"],
                            money=["strike", "preco", "custo", "preco_ativo", "valor_realista"],
                            num=["mult_realista"]),
                 hide_index=True, width="stretch", height=460)
    st.caption("**meta_hoje / meta_venc**: Realista (≥20% de chance), Possível (5–20%), Ousado (1–5%), Muito ousado (0,1–1%), Praticamente impossível (<0,1%). **valor_realista / mult_realista**: quanto o capital viraria nessa opção se a ação andar um movimento típico do dia a favor. "
               "**mov_hoje**: quanto a ação precisa andar até o fechamento de hoje para a opção valer a meta. "
               "**prob_hoje / prob_venc**: chance estimada (log-normal). **retorno_esperado**: valor teórico ÷ "
               "preço − 1 (negativo = opção cara). Preço = último negócio; confira o book. Sem custos e IR. "
               "**Zere a posição antes do vencimento: opção dentro do dinheiro no vencimento é exercida e vira compra ou venda de ações.**")


with top[0]:
    painel_monitor()
with top[1]:
    painel_risco()

# ================================================================ Outros menus (1 ativo)
hist, grade, vencs, erros = None, None, [], []

if fonte == "Demonstração":
    hist = dados.historico_demo()
    vencs = dados.vencimentos_demo()
elif fonte == "Online (gratuito)":
    try:
        vencs, v_padrao, grade_padrao = c_venc_grade(ticker)
    except Exception as e:
        erros.append(f"Não consegui listar vencimentos em opcoes.net.br: {e}")
    try:
        hist = c_hist(ticker)
    except Exception as e:
        erros.append(f"Histórico do Yahoo Finance indisponível: {e}")
else:
    arq = st.sidebar.file_uploader("CSV da grade de opções", type=["csv", "txt"],
                                   help="Colunas: codigo, tipo (CALL/PUT), strike, preco, negocios, vencimento")
    s_manual = st.sidebar.number_input("Preço atual do ativo (R$)", 0.0, 10000.0, 0.0, 0.01)
    if arq is not None:
        try:
            csv = dados.ler_csv(arq)
            for v in sorted(csv["vencimento"].dropna().unique()):
                vencs.append(dict(data=v, du=dias_uteis(date.today(), datetime.strptime(v, "%Y-%m-%d").date()),
                                  mensal=True, selecionado=False))
        except Exception as e:
            erros.append(f"Erro ao ler CSV: {e}")
    try:
        hist = c_hist(ticker)
    except Exception:
        pass

for e in erros:
    st.sidebar.warning(e)

if not vencs:
    OUT.title("Analisador de Opções B3")
    OUT.info("Sem vencimentos disponíveis. Verifique o ativo/conexão, envie um CSV ou use o modo Demonstração.")
    st.stop()


def rotulo(v):
    d = datetime.strptime(v["data"], "%Y-%m-%d").strftime("%d/%m/%Y")
    return f"{d}  ({v['du']} d.u.){'  • mensal' if v.get('mensal') else ''}"


padrao = next((i for i, v in enumerate(vencs) if v.get("selecionado")), 0)
venc = st.sidebar.selectbox("Vencimento", vencs, index=padrao, format_func=rotulo)
du = venc["du"] or dias_uteis(date.today(), datetime.strptime(venc["data"], "%Y-%m-%d").date())
du = max(int(du), 1)
T = du / DIAS_ANO

# grade do vencimento
try:
    if fonte == "Demonstração":
        S_demo = float(hist["Close"].iloc[-1])
        grade = dados.grade_demo(S_demo, venc["data"], r=r)
    elif fonte == "Online (gratuito)":
        if venc["data"] == v_padrao["data"]:
            grade = grade_padrao
        else:
            with OUT, st.spinner("Baixando grade de opções..."):
                grade = c_grade(ticker, venc["data"])
    else:
        grade = csv[csv["vencimento"] == venc["data"]].copy()
except Exception as e:
    OUT.error(f"Falha ao obter a grade de opções: {e}")
    st.stop()

# preço à vista: histórico > grade > manual
S = np.nan
if hist is not None and not hist.empty:
    S = float(hist["Close"].iloc[-1])
if fonte == "Online (gratuito)":
    s_grade = dados.spot_da_grade(grade)
    if np.isfinite(s_grade):
        S = s_grade  # a grade é mais atual que o fechamento do Yahoo
if fonte == "Arquivo CSV" and s_manual > 0:
    S = s_manual
if not np.isfinite(S) or S <= 0:
    OUT.error("Não foi possível determinar o preço do ativo. Informe-o manualmente (modo CSV).")
    st.stop()

# volatilidades históricas
hv = {}
if hist is not None and len(hist) > 30:
    for j in (21, 63, 252):
        s = vol_historica(hist["Close"], j).dropna()
        if len(s):
            hv[j] = float(s.iloc[-1])
hv21 = hv.get(21)

g = enriquecer(grade, S, du, r, hv21)
liq = g[(g["negocios"] >= min_neg) & g["iv"].notna()]

# IV ATM (média das duas opções mais próximas do dinheiro)
atm = liq.assign(d=(liq["strike"] - S).abs()).sort_values("d").head(4)
iv_atm = float(atm["iv"].median()) if not atm.empty else (hv21 or 0.3)

# ================================================================ cabeçalho
OUT.title(f"{ticker} — opções vencimento {datetime.strptime(venc['data'], '%Y-%m-%d'):%d/%m/%Y}")

c = OUT.columns(6)
c[0].metric("Preço do ativo", f"R$ {S:,.2f}")
c[1].metric("Dias úteis", du)
c[2].metric("Vol. implícita ATM", PCT(iv_atm))
c[3].metric("Vol. histórica 21d", PCT(hv21) if hv21 else "—")
c[4].metric("Vol. histórica 63d", PCT(hv[63]) if 63 in hv else "—")
if hv21:
    razao = iv_atm / hv21
    c[5].metric("IV / HV21", f"{razao:.2f}",
                "prêmios caros" if razao > 1.15 else ("prêmios baratos" if razao < 0.85 else "neutro"),
                delta_color="off")


aba = [None] + list(OUT.tabs(["Grade de opções", "Volatilidade", "Scanner de renda", "Scanner de travas",
                         "Montador de estratégias", "Alto risco (1 ativo)", "Como usar"]))

# ================================================================ 1. grade
with aba[1]:
    so_liq = st.checkbox("Mostrar só opções com liquidez", True)
    base = g[g["negocios"] >= min_neg] if so_liq else g
    cols = ["codigo", "vencimento", "strike", "preco", "negocios", "iv", "delta", "gama", "theta", "vega",
            "prob_exerc", "dist_pct", "valor_extr", "modelo"]
    l, rr = st.columns(2)
    for col, tipo in ((l, "CALL"), (rr, "PUT")):
        col.subheader(tipo + "s")
        col.dataframe(fmt_tabela(base[base["tipo"] == tipo][cols],
                                 pct=["iv", "prob_exerc", "dist_pct"],
                                 money=["strike", "preco", "valor_extr"],
                                 num=["delta", "gama", "theta", "vega"]),
                      hide_index=True, width="stretch", height=520)
    st.caption("Theta em R\\$ por dia útil; vega em R\\$ por 1 p.p. de volatilidade; "
               "prob. de exercício pelo modelo Black-Scholes (neutro ao risco).")

# ================================================================ 2. volatilidade
with aba[2]:
    l, rr = st.columns(2)
    with l:
        st.subheader("Sorriso de volatilidade")
        fig = go.Figure()
        for tipo, cor in (("CALL", AZUL), ("PUT", LARANJA)):
            x = liq[liq["tipo"] == tipo]
            fig.add_scatter(x=x["strike"], y=x["iv"], mode="markers+lines", name=tipo,
                            line=dict(color=cor, width=2), marker=dict(size=7),
                            customdata=x[["codigo", "negocios"]],
                            hovertemplate="%{customdata[0]}<br>K=%{x:.2f}<br>IV=%{y:.1%}"
                                          "<br>negócios=%{customdata[1]}<extra></extra>")
        fig.add_vline(x=S, line=dict(color=CINZA, dash="dot"), annotation_text="preço atual")
        if hv21:
            fig.add_hline(y=hv21, line=dict(color=CINZA, dash="dash"), annotation_text="HV 21d")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Strike", yaxis_title="Vol. implícita",
                          height=420, margin=dict(t=20, b=40), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    with rr:
        st.subheader("Volatilidade histórica")
        if hist is not None and len(hist) > 60:
            fig = go.Figure()
            for j, cor in ((21, AZUL), (63, LARANJA)):
                s = vol_historica(hist["Close"], j)
                fig.add_scatter(x=s.index, y=s, name=f"HV {j}d", line=dict(color=cor, width=2))
            fig.add_hline(y=iv_atm, line=dict(color=VERMELHO, dash="dash"), annotation_text="IV ATM hoje")
            fig.update_layout(yaxis_tickformat=".0%", height=420, margin=dict(t=20, b=40),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
            s = vol_historica(hist["Close"], 21).dropna().tail(252)
            pctil = float((s < iv_atm).mean())
            st.info(f"A IV ATM atual ({PCT(iv_atm)}) está acima de **{pctil:.0%}** das leituras de HV 21d "
                    f"do último ano. Acima de ~70%: ambiente favorece **vender** volatilidade (renda, travas "
                    f"de crédito). Abaixo de ~30%: favorece **comprar** (travas de débito, straddle).")
        else:
            st.info("Histórico indisponível para calcular a volatilidade histórica.")

    st.subheader("Opções mais caras e mais baratas vs. volatilidade histórica")
    if hv21 and not liq.empty:
        x = liq[["codigo", "vencimento", "tipo", "strike", "preco", "negocios", "iv", "iv_hv", "delta"]].sort_values("iv_hv")
        a, b = st.columns(2)
        a.caption("Mais baratas (IV/HV menor) — candidatas a compra")
        a.dataframe(fmt_tabela(x.head(10), pct=["iv"], money=["strike", "preco"], num=["iv_hv", "delta"]),
                    hide_index=True, width="stretch")
        b.caption("Mais caras (IV/HV maior) — candidatas a venda")
        b.dataframe(fmt_tabela(x.tail(10).iloc[::-1], pct=["iv"], money=["strike", "preco"],
                               num=["iv_hv", "delta"]), hide_index=True, width="stretch")

# ================================================================ 3. scanner de renda
with aba[3]:
    if so_opcoes:
        st.info("Venda coberta e venda de put para comprar a ação envolvem **ações**, então ficam ocultas no modo "
                "\"Só opções\". Desligue a opção na barra lateral para vê-las. Para operar só opções, use o "
                "**Scanner de travas** e o **Montador**.")
    else:
        st.subheader("Venda coberta e venda de put")
        f1, f2, f3 = st.columns(3)
        faixa_delta = f1.slider("|Delta| (≈ chance de exercício)", 0.0, 1.0, (0.15, 0.45), 0.05)
        ordem = f2.selectbox("Ordenar por", ["taxa_aa", "taxa", "prob_ficar", "protecao", "iv_hv"])
        tipo_r = f3.multiselect("Estratégia", ["Venda coberta", "Venda de put"],
                                ["Venda coberta", "Venda de put"])
        ren = scanner_renda(g, S, du, min_neg)
        ren = ren[ren["delta"].abs().between(*faixa_delta) & ren["estrategia"].isin(tipo_r)]
        ren = ren.sort_values(ordem, ascending=False)
        cols = ["estrategia", "codigo", "vencimento", "strike", "preco", "negocios", "dist_pct", "taxa", "taxa_aa",
                "retorno_exercido", "protecao", "preco_efetivo", "prob_ficar", "iv", "iv_hv", "delta"]
        st.dataframe(fmt_tabela(ren[cols], pct=["dist_pct", "taxa", "taxa_aa", "retorno_exercido", "protecao",
                                                "prob_ficar", "iv"],
                                money=["strike", "preco", "preco_efetivo"], num=["iv_hv", "delta"]),
                     hide_index=True, width="stretch", height=480)
        selic_per = (1 + selic / 100) ** (du / DIAS_ANO) - 1
        st.caption(f"**taxa**: valor extrínseco / capital (o que você ganha se o preço não mudar). "
                   f"**taxa_aa**: anualizada — compare com a Selic de {selic:.2f}% a.a. "
                   f"(≈ {selic_per:.2%} no período). **proteção**: queda que o prêmio absorve (venda coberta) "
                   f"ou desconto do preço de compra efetivo vs. hoje (put). **prob_ficar**: chance, pelo modelo, "
                   f"de a opção virar pó e você ficar com o prêmio.")

# ================================================================ 4. scanner de travas
with aba[4]:
    st.subheader("Travas verticais")
    f1, f2, f3, f4 = st.columns(4)
    base_vol = f1.selectbox("Volatilidade para probabilidades",
                            ["IV ATM (mercado)", "HV 21d (histórica)", "Manual"],
                            help="Com a IV do mercado o valor esperado tende a zero. "
                                 "Use a sua previsão de vol (HV ou manual) para achar travas com vantagem.")
    if base_vol.startswith("IV"):
        sig_t = iv_atm
    elif base_vol.startswith("HV"):
        sig_t = hv21 or iv_atm
    else:
        sig_t = f1.number_input("Vol. (% a.a.)", 1.0, 300.0, round(iv_atm * 100, 1)) / 100
    faixa = f2.slider("Strikes até ±% do preço", 5, 40, 15) / 100
    ord_t = f3.selectbox("Ordenar por", ["valor_esperado", "ve_por_risco", "prob_lucro", "ganho_risco"])
    tipos_t = f4.multiselect("Tipos", ["Trava de alta (call, débito)", "Trava de alta (put, crédito)",
                                       "Trava de baixa (put, débito)", "Trava de baixa (call, crédito)"],
                             ["Trava de alta (call, débito)", "Trava de alta (put, crédito)",
                              "Trava de baixa (put, débito)", "Trava de baixa (call, crédito)"])
    with st.spinner("Varrendo combinações..."):
        tr = scanner_travas(g, S, du, r, sig_t, min_neg, faixa)
    if tr.empty:
        st.info("Nenhuma trava encontrada com os filtros atuais (tente reduzir o mínimo de negócios).")
    else:
        tr = tr[tr["estrategia"].isin(tipos_t)].sort_values(ord_t, ascending=False)
        tr.insert(3, "vencimento", venc["data"])
        st.caption(f"{len(tr)} combinações · vol usada: {PCT(sig_t)} · valores por unidade (×100 por lote)")
        st.dataframe(fmt_tabela(tr.head(200), pct=["prob_lucro"],
                                money=["k_menor", "k_maior", "largura", "fluxo_inicial", "ganho_max",
                                       "perda_max", "valor_esperado"], num=["ganho_risco", "ve_por_risco"]),
                     hide_index=True, width="stretch", height=480)
        st.caption("**fluxo_inicial**: positivo = você recebe (crédito), negativo = você paga (débito). "
                   "**valor_esperado**: resultado médio pela distribuição log-normal com a vol escolhida. "
                   "Preços são o *último negócio*; confira o book (compra/venda) antes de operar.")

# ================================================================ 5. montador
MODELOS = {
    "Venda coberta": [("ACAO", 1, 0), ("CALL", -1, 0.05)],
    "Venda de put": [("PUT", -1, -0.05)],
    "Trava de alta com call": [("CALL", 1, 0), ("CALL", -1, 0.06)],
    "Trava de baixa com put": [("PUT", 1, 0), ("PUT", -1, -0.06)],
    "Straddle comprado": [("CALL", 1, 0), ("PUT", 1, 0)],
    "Strangle vendido": [("CALL", -1, 0.08), ("PUT", -1, -0.08)],
    "Borboleta com call": [("CALL", 1, -0.05), ("CALL", -2, 0), ("CALL", 1, 0.05)],
    "Iron condor": [("PUT", 1, -0.12), ("PUT", -1, -0.06), ("CALL", -1, 0.06), ("CALL", 1, 0.12)],
    "Compra a seco de call": [("CALL", 1, 0.03)],
    "Compra de put (seguro)": [("ACAO", 1, 0), ("PUT", 1, -0.05)],
}


def montar_modelo(nome):
    pernas = []
    for tipo, q, desloc in MODELOS[nome]:
        if tipo == "ACAO":
            pernas.append(dict(tipo="ACAO", qtd=q, codigo=ticker, strike=None, premio=round(S, 2), iv=None))
            continue
        x = liq[liq["tipo"] == tipo]
        if x.empty:
            x = g[(g["tipo"] == tipo) & (g["preco"] > 0)]
        if x.empty:
            continue
        o = x.loc[(x["strike"] - S * (1 + desloc)).abs().idxmin()]
        pernas.append(dict(tipo=tipo, qtd=q, codigo=o["codigo"], strike=float(o["strike"]),
                           premio=float(o["preco"]), iv=o["iv"]))
    return pd.DataFrame(pernas)


with aba[5]:
    st.subheader("Montador de estratégias")
    a, b, c3 = st.columns([2, 1, 1])
    if so_opcoes:
        MODELOS = {k_: v_ for k_, v_ in MODELOS.items() if all(p_[0] != "ACAO" for p_ in v_)}
    modelo = a.selectbox("Modelo inicial", list(MODELOS))
    lotes = b.number_input("Lotes (×100)", 1, 1000, 1)
    sig_m = c3.number_input("Vol. p/ probabilidades (%)", 1.0, 300.0, round(iv_atm * 100, 1)) / 100
    if so_opcoes:
        st.warning("Zere a posição antes do vencimento: opção dentro do dinheiro no vencimento é exercida e vira compra ou venda de ações. Posições vendidas (strangle, venda de put, travas de crédito) também podem ser exercidas "
                   "antes, se forem americanas, e exigem margem na corretora.")
    st.caption("Edite as pernas à vontade: qtd positiva = compra, negativa = venda. "
               "Troque o strike/prêmio ou adicione linhas.")
    chave = f"pernas_{modelo}_{venc['data']}_{ticker}"
    ed = st.data_editor(
        montar_modelo(modelo), key=chave, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "tipo": st.column_config.SelectboxColumn("tipo", options=["CALL", "PUT"] if so_opcoes else ["CALL", "PUT", "ACAO"], required=True),
            "qtd": st.column_config.NumberColumn("qtd", step=1),
            "strike": st.column_config.NumberColumn("strike", format="%.2f"),
            "premio": st.column_config.NumberColumn("prêmio / preço", format="%.2f"),
            "iv": st.column_config.NumberColumn("iv", format="%.3f", disabled=True),
        })
    pernas = [p for p in ed.to_dict("records") if p.get("tipo") and pd.notna(p.get("premio")) and pd.notna(p.get("qtd"))
              and (p["tipo"] == "ACAO" or pd.notna(p.get("strike")))]
    if pernas:
        m = metricas(pernas, S, T, r, sig_m)
        mult = 100 * lotes
        fmt_r = lambda v: "ilimitado" if not np.isfinite(v) else f"R$ {v * mult:,.2f}"
        k = st.columns(5)
        k[0].metric("Prêmios (líquido)", f"R$ {-m['custo_montagem'] * mult:,.2f}",
                    "crédito" if m["custo_montagem"] < 0 else "débito", delta_color="off")
        k[1].metric("Ganho máximo", fmt_r(m["ganho_max"]))
        k[2].metric("Perda máxima", fmt_r(m["perda_max"]))
        k[3].metric("Prob. de lucro", PCT(m["prob_lucro"]))
        k[4].metric("Valor esperado", f"R$ {m['valor_esperado'] * mult:,.2f}")
        if m["breakevens"]:
            st.caption("Ponto(s) de equilíbrio: " + ", ".join(f"R\\$ {br(b)} ({b / S - 1:+.1%})"
                                                                for b in m["breakevens"]))
        lo, hi = S * 0.7, S * 1.3
        sel = (m["grade"] >= lo) & (m["grade"] <= hi)
        xg = m["grade"][sel]
        yv = m["payoff"][sel] * mult
        yh = payoff_hoje(pernas, xg, T, r, sig_m) * mult
        fig = go.Figure()
        fig.add_scatter(x=xg, y=np.where(yv >= 0, yv, np.nan), fill="tozeroy", mode="none",
                        fillcolor="rgba(46,158,91,0.15)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=xg, y=np.where(yv < 0, yv, np.nan), fill="tozeroy", mode="none",
                        fillcolor="rgba(214,69,69,0.15)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=xg, y=yv, name="No vencimento", line=dict(color=AZUL, width=3),
                        hovertemplate="S=R$ %{x:.2f}<br>Resultado=R$ %{y:,.2f}<extra></extra>")
        fig.add_scatter(x=xg, y=yh, name="Hoje (teórico)", line=dict(color=LARANJA, width=2, dash="dash"),
                        hovertemplate="S=R$ %{x:.2f}<br>Resultado=R$ %{y:,.2f}<extra></extra>")
        fig.add_hline(y=0, line=dict(color=CINZA, width=1))
        fig.add_vline(x=S, line=dict(color=CINZA, dash="dot"), annotation_text="preço atual")
        s1 = S * np.exp(sig_m * np.sqrt(T))
        s0 = S * np.exp(-sig_m * np.sqrt(T))
        fig.add_vrect(x0=s0, x1=s1, fillcolor=CINZA, opacity=0.08, line_width=0,
                      annotation_text="±1 desvio", annotation_position="top left")
        fig.update_layout(height=460, xaxis_title=f"Preço de {ticker} no vencimento",
                          yaxis_title="Resultado (R$)", margin=dict(t=30, b=40),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

        # gregas agregadas
        tot = dict(delta=0.0, gama=0.0, theta=0.0, vega=0.0)
        for p in pernas:
            if p["tipo"] == "ACAO":
                tot["delta"] += p["qtd"]
                continue
            linha = g[(g["tipo"] == p["tipo"]) & (np.isclose(g["strike"], float(p["strike"])))]
            if not linha.empty:
                for k_ in tot:
                    v = linha.iloc[0][k_]
                    tot[k_] += p["qtd"] * (v if np.isfinite(v) else 0)
        k = st.columns(4)
        k[0].metric("Delta da posição", f"{tot['delta'] * mult:,.0f} ações")
        k[1].metric("Gama", f"{tot['gama'] * mult:,.1f}")
        k[2].metric("Theta (R$/dia)", f"R$ {tot['theta'] * mult:,.2f}")
        k[3].metric("Vega (R$/1 p.p.)", f"R$ {tot['vega'] * mult:,.2f}")


# ================================================================ 6. alto risco
with aba[6]:
    st.subheader("Alto risco — ativo da barra lateral")
    st.error("Compra de opções fora do dinheiro: o resultado mais provável é **perder 100% do valor aplicado**. "
             "Use apenas dinheiro que você pode perder sem prejudicar suas contas.")
    f1, f2, f3, f4 = st.columns(4)
    capital, meta = capital_g, meta_g
    f1.metric("Capital", f"R$ {capital:,.2f}")
    f2.metric("Meta", f"R$ {meta:,.2f}")
    mult = meta / capital
    base_v = f3.selectbox("Vol. para probabilidades", ["HV 21d (histórica)", "IV ATM (mercado)"], key="vol_ar")
    sig_ar = (hv21 or iv_atm) if base_v.startswith("HV") else iv_atm
    tipo_ar = f4.multiselect("Tipo", ["CALL", "PUT"], ["CALL", "PUT"], key="tipo_ar")
    st.caption(f"Multiplicador necessário: **{mult:.1f}x** · vol usada: {PCT(sig_ar)} · "
               f"movimento típico de 1 dia: ±{sig_ar / np.sqrt(DIAS_ANO):.1%}")
    ar = scanner_multiplicador(g, S, du, r, sig_ar, capital, mult, min_neg)
    if ar.empty:
        st.info("Sem opções líquidas para analisar neste vencimento.")
    else:
        ar = ar[ar["tipo"].isin(tipo_ar) & (ar["qtd"] > 0)].sort_values("prob_hoje", ascending=False)
        if ar.empty:
            st.warning("Com esse capital não dá para comprar nem um lote (100 opções) de nenhuma série líquida.")
        else:
            best_h, best_v = ar["prob_hoje"].max(), ar["prob_venc"].max()
            k = st.columns(3)
            k[0].metric(f"Melhor chance de {mult:.1f}x HOJE", f"{best_h:.2%}")
            k[1].metric(f"Melhor chance de {mult:.1f}x até o vencimento", f"{best_v:.2%}")
            k[2].metric("Retorno esperado médio (pelo modelo)", f"{ar['retorno_esperado'].median():+.0%}")
            um_em = f"≈ 1 em {1 / best_h:,.0f}" if best_h > 0 else "praticamente nula"
            st.info(f"Na melhor opção, a chance de bater a meta hoje é {um_em}. "
                    f"Ou seja: se você fizesse essa aposta muitas vezes, na grande maioria perderia o capital inteiro.")
            ar["vencimento"] = venc["data"]
            cols = ["codigo", "vencimento", "meta_hoje", "valor_realista", "mult_realista", "tipo", "strike", "preco",
                    "negocios", "qtd", "custo", "mov_hoje", "prob_hoje", "meta_venc", "mov_venc", "prob_venc",
                    "prob_lucro_venc", "retorno_esperado", "iv"]
            st.dataframe(fmt_tabela(ar[cols], pct=["mov_hoje", "prob_hoje", "mov_venc", "prob_venc",
                                                   "prob_lucro_venc", "retorno_esperado", "iv"],
                                    money=["strike", "preco", "custo", "valor_realista"], num=["mult_realista"]),
                         hide_index=True, width="stretch", height=460)
            st.caption("**meta_hoje / meta_venc**: Realista (≥20% de chance), Possível (5–20%), Ousado (1–5%), Muito ousado (0,1–1%), Praticamente impossível (<0,1%). **valor_realista / mult_realista**: quanto o capital viraria nessa opção se a ação andar um movimento típico do dia a favor. "
                       "**mov_hoje**: quanto a ação precisa subir (call) ou cair (put) até o fechamento de hoje "
                       "para a opção valer a meta, mantendo a IV. **prob_hoje / prob_venc**: chance estimada pela "
                       "distribuição log-normal. **prob_lucro_venc**: chance de terminar acima do custo no vencimento. "
                       "**retorno_esperado**: valor teórico com a vol escolhida ÷ preço − 1 (negativo = opção cara). "
                       "Lote mínimo de 100 opções. Não considera spread do book, corretagem nem IR.")

# ================================================================ 7. ajuda
with aba[7]:
    st.markdown("""
### Fluxo sugerido
1. **Volatilidade** — veja se a IV está cara ou barata vs. a histórica. Isso define o lado:
   IV alta → vender prêmio (venda coberta, venda de put, travas de crédito, condor);
   IV baixa → comprar (travas de débito, straddle).
2. **Scanner de renda** — para quem já tem a ação (venda coberta) ou aceita comprá-la mais barato (venda de put).
   Compare a *taxa_aa* com a Selic: se não pagar bem acima do CDI, o risco não compensa.
3. **Scanner de travas** — escolha a vol que *você* acredita e ordene por valor esperado.
   Com a IV do mercado o valor esperado fica perto de zero (o mercado é eficiente em média);
   vantagem só aparece se a sua previsão de vol ou direção for melhor que a do mercado.
4. **Montador** — refine a operação, veja payoff, pontos de equilíbrio e gregas antes de enviar a ordem.

### Cuidados importantes
- O preço usado é o **último negócio**, que pode estar desatualizado. Sempre confira o book de ofertas.
- Opções americanas (modelo **A**) podem ser exercidas antes; o modelo aqui é europeu (Black-Scholes).
- Venda a seco de call tem **perda ilimitada**; venda de put exige garantia (o valor do strike × quantidade).
- Corretagem, emolumentos B3 e **IR (15% swing / 20% day trade)** reduzem o resultado.
- Gerencie o tamanho: não arrisque numa operação mais do que 1–2% do patrimônio.

### Importar CSV
Colunas aceitas: `codigo, tipo, strike, preco, negocios, vencimento` (separador `;` ou `,`,
decimal com vírgula ou ponto, data `dd/mm/aaaa` ou `aaaa-mm-dd`).
""")
    with st.expander("Diagnóstico dos dados"):
        st.write(f"Fonte: {fonte} · opções no vencimento: {len(g)} · com liquidez: {len(liq)}")
        st.dataframe(grade.head(10), width="stretch")
