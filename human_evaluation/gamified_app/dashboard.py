"""
dashboard.py — Interactive results viewer for the WoonPraat Gamified Human Eval.

Reads results.xlsx (written live by main.py) and item_bank.json, and renders two
views: Rate & Reveal (score distributions, agreement, per-item table) and Spot
the Fake (accuracy, confusion matrix). Charts are Plotly, so hover for detail
and click points in the scatter to inspect the underlying post/comment.

Run with:
    streamlit run dashboard.py
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(_HERE, "results.xlsx")
ITEM_BANK_PATH = os.path.join(_HERE, "item_bank.json")
METRICS = ["fluency", "relevance", "humanness"]

st.set_page_config(page_title="WoonPraat Results", layout="wide")


@st.cache_data(ttl=30)
def load_data():
    with open(ITEM_BANK_PATH, encoding="utf-8") as f:
        bank_df = pd.DataFrame(json.load(f))[["item_id", "post", "comment"]]

    rate_df = pd.read_excel(RESULTS_PATH, sheet_name="rate_and_reveal")
    spot_df = pd.read_excel(RESULTS_PATH, sheet_name="spot_the_fake")
    rate_df = rate_df.merge(bank_df, on="item_id", how="left")
    spot_df = spot_df.merge(bank_df, on="item_id", how="left")
    return rate_df, spot_df


def excerpt(text, n=90):
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= n else text[:n].rstrip() + "…"


top = st.columns([5, 1])
top[0].title("WoonPraat results")
if top[1].button("Refresh data"):
    st.cache_data.clear()

if not os.path.exists(RESULTS_PATH):
    st.info("No results.xlsx yet — play a round first, then refresh.")
    st.stop()

rate_df, spot_df = load_data()

tab_rate, tab_spot = st.tabs(["Rate & Reveal", "Spot the Fake"])

# ── Rate & Reveal ──────────────────────────────────────────────────────
with tab_rate:
    if rate_df.empty:
        st.info("No Rate & Reveal submissions yet.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ratings", len(rate_df))
        k2.metric("Items rated", rate_df["item_id"].nunique())
        k3.metric("Sessions", rate_df["session_id"].nunique())
        k4.metric("Participants", rate_df["user_id"].replace("", pd.NA).nunique())

        raters_per_item = rate_df.groupby("item_id").size()
        overlap_items = raters_per_item[raters_per_item >= 2].index

        # Facilitated sessions carry a user_id; free-play ones don't, so fall
        # back to a short session_id so every rating still has a rater label.
        annotator = rate_df["user_id"].astype(str).str.strip()
        rate_df["annotator"] = annotator.mask(
            annotator.eq("") | annotator.eq("nan"), rate_df["session_id"].str[:8]
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Score distribution by condition")
            metric = st.radio("Metric", METRICS, horizontal=True, key="dist_metric")
            fig = px.box(
                rate_df, x="condition", y=metric, color="condition",
                points="all", hover_data=["item_id"],
                category_orders={"condition": ["real", "synthetic"]},
            )
            fig.update_layout(showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Mean score ± std, by condition")
            summary = rate_df.groupby("condition")[METRICS].agg(["mean", "std"])
            fig = go.Figure()
            for m in METRICS:
                fig.add_bar(
                    name=m,
                    x=summary.index,
                    y=summary[(m, "mean")],
                    error_y=dict(type="data", array=summary[(m, "std")]),
                )
            fig.update_layout(barmode="group", yaxis_range=[1, 5], margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fluency vs. relevance, colored by humanness (click a point for the full text)")
        scatter = px.scatter(
            rate_df, x="fluency", y="relevance",
            color="humanness", color_continuous_scale="Teal",
            symbol="condition", symbol_map={"real": "circle", "synthetic": "diamond"},
            hover_data=["item_id"],
            category_orders={"condition": ["real", "synthetic"]},
        )
        scatter.update_traces(marker=dict(size=11, line=dict(width=1, color="rgba(0,0,0,0.25)")))
        scatter.update_layout(margin=dict(t=10))
        event = st.plotly_chart(
            scatter, use_container_width=True, on_select="rerun", key="fluency_scatter"
        )
        picks = event.selection.points if event and event.selection else []
        if picks:
            # color="condition" splits the scatter into per-condition traces, so
            # point_indices are positions within a trace, not into rate_df — read
            # the item_id back out of each point's own data instead.
            picked_ids = {p["customdata"][0] for p in picks}
            for item_id in picked_ids:
                row = rate_df[rate_df["item_id"] == item_id].iloc[0]
                st.markdown(
                    f"**Item {row['item_id']}** · `{row['condition']}` · "
                    f"fluency {row['fluency']}, relevance {row['relevance']}, humanness {row['humanness']}"
                )
                st.markdown(f"> {row['post']}")
                st.markdown(f"— {row['comment']}")
        else:
            st.caption("No point selected yet.")

        st.subheader("Compare annotators")
        annotators = sorted(rate_df["annotator"].unique())
        chosen = st.multiselect("Annotators", annotators, default=annotators, key="cmp_annotators")
        cmp_df = rate_df[rate_df["annotator"].isin(chosen)].copy()

        heat_metric = st.radio("Metric", METRICS, horizontal=True, key="heat_metric")
        st.caption(f"{heat_metric.capitalize()} per item, by annotator — item labels end in (R)eal / (S)ynthetic; "
                   f"blank means that annotator didn't rate that item")

        # Label columns with condition (R/S) instead of the bare item_id, and keep
        # them ordered by item_id — pivot_table would otherwise sort labels as text.
        cmp_df["item_label"] = cmp_df["item_id"].astype(str) + " (" + cmp_df["condition"].str[0].str.upper() + ")"
        label_order = (cmp_df[["item_id", "item_label"]]
                        .drop_duplicates()
                        .sort_values("item_id")["item_label"])

        pivot = cmp_df.pivot_table(index="annotator", columns="item_label", values=heat_metric)
        pivot = pivot.reindex(index=chosen, columns=label_order)
        fig = px.imshow(
            pivot, text_auto=True, color_continuous_scale="Teal", aspect="auto",
            labels=dict(x="item (condition)", y="annotator", color=heat_metric),
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        if len(overlap_items):
            st.subheader(f"Rater agreement ({len(overlap_items)} items with 2+ raters)")
            overlap_df = rate_df[rate_df["item_id"].isin(overlap_items)]
            agree_rows = []
            for m in METRICS:
                spread = overlap_df.groupby("item_id")[m].agg(lambda s: s.max() - s.min())
                agree_rows.append({"metric": m, "mean_abs_diff": round(spread.mean(), 2),
                                    "exact_match_rate": round((spread == 0).mean(), 2)})
            st.dataframe(pd.DataFrame(agree_rows), hide_index=True, use_container_width=True)

        st.subheader("Per-item summary")
        item_table = (
            rate_df.groupby(["item_id", "condition"])
            .agg(n_raters=("session_id", "count"), **{m: (m, "mean") for m in METRICS})
            .reset_index()
            .merge(rate_df[["item_id", "post"]].drop_duplicates(), on="item_id")
        )
        item_table["post"] = item_table["post"].apply(excerpt)
        st.dataframe(item_table.sort_values("item_id"), hide_index=True, use_container_width=True)

# ── Spot the Fake ──────────────────────────────────────────────────────
with tab_spot:
    if spot_df.empty:
        st.info("No Spot the Fake submissions yet.")
    else:
        acc = spot_df["correct"].mean()
        k1, k2, k3 = st.columns(3)
        k1.metric("Rounds played", len(spot_df))
        k2.metric("Overall accuracy", f"{acc:.0%}")
        k3.metric("Sessions", spot_df["session_id"].nunique())

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Accuracy by condition")
            by_cond = spot_df.groupby("condition")["correct"].mean().reindex(["real", "synthetic"])
            fig = px.bar(by_cond, y=by_cond.values, x=by_cond.index, labels={"y": "accuracy", "x": "condition"})
            fig.update_layout(yaxis_range=[0, 1], margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Guess vs. actual condition")
            confusion = pd.crosstab(spot_df["condition"], spot_df["user_guess"])
            fig = px.imshow(
                confusion, text_auto=True, color_continuous_scale="Teal",
                labels=dict(x="guessed", y="actual", color="count"),
            )
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Accuracy over time")
        ts = spot_df.sort_values("timestamp").copy()
        ts["rolling_accuracy"] = ts["correct"].expanding().mean()
        fig = px.line(ts, x="timestamp", y="rolling_accuracy")
        fig.update_layout(yaxis_range=[0, 1], margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-item accuracy")
        item_table = (
            spot_df.groupby(["item_id", "condition"])
            .agg(n_guesses=("session_id", "count"), accuracy=("correct", "mean"))
            .reset_index()
            .merge(spot_df[["item_id", "post"]].drop_duplicates(), on="item_id")
        )
        item_table["post"] = item_table["post"].apply(excerpt)
        st.dataframe(item_table.sort_values("item_id"), hide_index=True, use_container_width=True)
