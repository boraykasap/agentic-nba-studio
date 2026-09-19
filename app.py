#!/usr/bin/env python3
"""
NBA Studio

Two pages:
  1. Customer   - pick a client, see their enriched profile + precomputed NBAs,
                  chat with a supervisor-style assistant grounded in their data.
  2. Campaigns  - browse the target audience for an existing campaign, or describe
                  a new one and get a ready-to-target audience.

Usage:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="NBA Studio", layout="wide")

# Team branding, rendered before st.navigation()/pg.run() so it sits at the top of the main
# content area (right-aligned) above every page, rather than in the sidebar.
header_cols = st.columns([6, 1])
with header_cols[1]:
    st.markdown(
        "<div style='text-align:right; font-weight:600; font-size:1.1rem;'>NextBext</div>",
        unsafe_allow_html=True,
    )

customer_page = st.Page("views/customer.py", title="Customer", default=True)
query_page = st.Page("views/query.py", title="Campaigns")

pg = st.navigation([customer_page, query_page])
pg.run()
