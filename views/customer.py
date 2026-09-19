import streamlit as st

from common import (
    load_enriched_df,
    get_enriched_client,
    get_client_context,
    client_label,
    render_profile_card,
    render_nba_cards,
    render_client_data_tables,
    build_system_prompt,
    chat_reply,
    refresh_recommendation,
)

st.title("NBA Studio — Customer")
st.markdown("Pick a client. See their profile + precomputed NBAs. Chat with a supervisor-style assistant grounded in their data.")

df = load_enriched_df()

options = df['individual_id'].tolist()
labels = {row['individual_id']: client_label(row) for row in df.to_dict('records')}

# If the Query page sent us here for a specific client, jump straight to them
jump_to = st.session_state.pop('jump_to_individual_id', None)
default_index = options.index(jump_to) if jump_to in options else 0

selected_id = st.selectbox(
    "Select a client",
    options=options,
    index=default_index,
    format_func=lambda x: labels.get(x, x),
)

if not selected_id:
    st.stop()

result = get_enriched_client(selected_id, df)
if result is None:
    st.error("Client not found in individual_nba_enriched.csv.")
    st.stop()
profile, nbas = result

with st.spinner("Loading additional client context from client data..."):
    client_context = get_client_context(selected_id)

render_profile_card(profile)

st.markdown("#### Precomputed NBA predictions")
render_nba_cards(nbas)

with st.expander("Additional context (client data: accounts, balances, recent events/transactions)"):
    render_client_data_tables(client_context, individual_id=selected_id)

st.divider()
st.subheader("Chat about this client")

system_prompt = build_system_prompt(profile, nbas, client_context)

# Each client keeps its own conversation + live recommendation, independent of the others,
# so switching clients (or pages) never loses a thread already in progress.
if "chats" not in st.session_state:
    st.session_state.chats = {}

if selected_id not in st.session_state.chats:
    st.session_state.chats[selected_id] = {
        "history": [],
        "live_recommendation": {
            "recommendation_name": nbas[0]['name'] if nbas else None,
            "rationale": nbas[0]['why'] if nbas else None,
            "changed_from_default": False,
            "change_reason": None,
        },
    }

chat_state = st.session_state.chats[selected_id]
rec = chat_state["live_recommendation"]
rec_box = st.container()

for msg in chat_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about this client (e.g. \"why lead with NBA #1?\")")
if user_input:
    chat_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    llm_messages = [{"role": "system", "content": system_prompt}] + chat_state["history"]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = chat_reply(llm_messages)
        st.markdown(reply)
    chat_state["history"].append({"role": "assistant", "content": reply})

    with st.spinner("Re-checking recommendation..."):
        chat_state["live_recommendation"] = refresh_recommendation(nbas, chat_state["history"])
    st.rerun()

with rec_box:
    if rec["changed_from_default"]:
        st.warning(
            f"**Updated recommendation: {rec['recommendation_name']}**\n\n"
            f"{rec.get('rationale', '')}\n\n"
            f"_Changed because: {rec.get('change_reason', 'new information in the conversation')}_"
        )
    else:
        st.info(f"**Current recommendation: {rec['recommendation_name']}**")
