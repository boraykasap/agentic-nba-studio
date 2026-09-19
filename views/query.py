import json

import numpy as np
import pandas as pd
import streamlit as st

from common import (
    load_enriched_df,
    get_action_catalog,
    client_label,
    fmt_chf,
    fmt_confidence,
    RANK_BADGE,
    load_customer_embeddings,
    campaign_reply,
    apply_hard_filters,
    score_customers_by_weights,
    EMBEDDING_DIMS,
)

st.title("NBA Studio — Campaigns")

df = load_enriched_df()

tab_action, tab_campaign = st.tabs(["Existing Campaigns", "New Campaign"])

# ============================================================================
# TAB 1: filter by one of the precomputed NBA actions
# ============================================================================

with tab_action:
    st.markdown("Pick a campaign to see its current target audience — every customer it's recommended for right now.")

    actions = get_action_catalog(df)

    if not actions:
        st.info("No campaigns found in the dataset.")
    else:
        action = st.selectbox(
            "Campaign",
            options=actions,
            format_func=lambda a: a['name'],
        )

        filter_cols = st.columns([2, 2])
        with filter_cols[0]:
            rank_filter = st.multiselect(
                "Include priority",
                options=[1, 2, 3],
                default=[1, 2, 3],
                format_func=lambda r: f"{RANK_BADGE.get(r, '')} #{r}",
                key="action_rank_filter",
            )
        with filter_cols[1]:
            min_confidence = st.slider("Minimum match score", 0.0, 1.0, 0.0, 0.05, key="action_min_conf")

        rows = []
        for r in rank_filter:
            id_col, why_col, conf_col = f'nba_{r}_id', f'nba_{r}_why', f'nba_{r}_confidence'
            matches = df[df[id_col] == action['id']]
            for rec in matches.to_dict('records'):
                rows.append({
                    'individual_id': rec['individual_id'],
                    'name': client_label(rec),
                    'rank': r,
                    'confidence': rec.get(conf_col),
                    'why': rec.get(why_col),
                    'city': rec.get('city'),
                    'canton': rec.get('canton'),
                    'age': rec.get('age'),
                    'income_chf': rec.get('income_chf'),
                })

        rows = [r for r in rows if pd.isna(r['confidence']) or float(r['confidence']) >= min_confidence]

        st.divider()

        if not rows:
            st.info(f"No customers currently qualify for **{action['name']}** at the selected priority/match score.")
        else:
            result_df = pd.DataFrame(rows).sort_values(['rank', 'confidence'], ascending=[True, False]).reset_index(drop=True)
            st.metric("Target audience size", len(result_df))

            for row in result_df.to_dict('records'):
                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        conf_label = fmt_confidence(row['confidence'])
                        header = f"**{RANK_BADGE.get(row['rank'], '')} {row['name']}** · priority #{row['rank']}"
                        if conf_label:
                            header += f" · match score {conf_label}"
                        st.markdown(header)
                        st.caption(
                            f"{row['city']}, {row['canton']} · {row['age']} y/o · income {fmt_chf(row['income_chf'])}"
                        )
                        if row['why'] and str(row['why']) != 'nan':
                            st.write(row['why'])
                    with cols[1]:
                        if st.button("View client →", key=f"view_action_{row['individual_id']}_{row['rank']}"):
                            st.session_state['jump_to_individual_id'] = row['individual_id']
                            st.switch_page("views/customer.py")

# ============================================================================
# TAB 2: new product/campaign -> chat with an assistant that proposes targeting
# criteria (hard eligibility filters + soft psychographic weights) and ranks customers
# ============================================================================

with tab_campaign:
    st.markdown(
        "Describe a new product or campaign in plain language, and get a ready-to-target audience — "
        "eligibility rules plus a customer-fit score — that you can refine by chatting."
    )

    if "campaign_llm_history" not in st.session_state:
        st.session_state.campaign_llm_history = []
        st.session_state.campaign_display_history = []
        st.session_state.campaign_hard_filters = []
        st.session_state.campaign_dimension_weights = {}

    for msg in st.session_state.campaign_display_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "e.g. \"new travel rewards credit card for young, sociable clients who don't already have a card\"",
        key="campaign_chat_input",
    )
    if user_input:
        st.session_state.campaign_display_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.campaign_llm_history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Working out who fits..."):
                result = campaign_reply(st.session_state.campaign_llm_history)
            st.markdown(result["reply"])
        st.session_state.campaign_llm_history.append({"role": "assistant", "content": json.dumps(result)})
        st.session_state.campaign_display_history.append({"role": "assistant", "content": result["reply"]})
        st.session_state.campaign_hard_filters = result["hard_filters"]
        st.session_state.campaign_dimension_weights = result["dimension_weights"]

    hard_filters = st.session_state.campaign_hard_filters
    dimension_weights = st.session_state.campaign_dimension_weights

    if hard_filters or dimension_weights:
        st.divider()
        st.markdown("#### Campaign targeting criteria")

        if hard_filters:
            st.caption("Eligibility rules")
            st.markdown("\n".join(f"- `{f['column']}` {f['op']} `{f['value']}`" for f in hard_filters))
        if dimension_weights:
            dim_labels = dict(EMBEDDING_DIMS)
            st.caption("Customer profile this campaign favors")
            chips = []
            for key, w in sorted(dimension_weights.items(), key=lambda kv: -abs(kv[1])):
                arrow = "↑" if w > 0 else "↓"
                chips.append(f"{arrow} {dim_labels.get(key, key)} ({w:+.2f})")
            st.markdown(" · ".join(chips))

        embeddings, id_to_row = load_customer_embeddings()
        mask, _ = apply_hard_filters(df, hard_filters)
        eligible_df = df[mask].reset_index(drop=True)

        if eligible_df.empty:
            st.info("No customers meet the current eligibility rules.")
        else:
            scores, _ = score_customers_by_weights(embeddings, dimension_weights)
            row_indices = np.array([id_to_row[i] for i in eligible_df['individual_id']])
            eligible_df = eligible_df.assign(_fit_score=scores[row_indices])
            eligible_df = eligible_df.sort_values('_fit_score', ascending=False).reset_index(drop=True)

            top_n = 20
            st.metric("Target audience size", len(eligible_df))
            st.caption(f"Showing top {min(top_n, len(eligible_df))} best-fit customers" if dimension_weights else f"Showing first {min(top_n, len(eligible_df))} eligible customers")

            for rec in eligible_df.head(top_n).to_dict('records'):
                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"**{client_label(rec)}**")
                        st.caption(
                            f"{rec.get('city')}, {rec.get('canton')} · {rec.get('age')} y/o · "
                            f"income {fmt_chf(rec.get('income_chf'))}"
                        )
                        if rec.get('nba_1_name') and str(rec.get('nba_1_name')) != 'nan':
                            conf = fmt_confidence(rec.get('nba_1_confidence'))
                            extra = f" ({conf})" if conf else ""
                            st.caption(f"Current top offer: {rec['nba_1_name']}{extra}")
                    with cols[1]:
                        if st.button("View client →", key=f"view_campaign_{rec['individual_id']}"):
                            st.session_state['jump_to_individual_id'] = rec['individual_id']
                            st.switch_page("views/customer.py")
