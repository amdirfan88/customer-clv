"""Streamlit presentation layer; all recommendations come from the agent.

This script only displays the conversation. Every product choice and every reply
comes from RecommendationAgent in b_recommendation_agent.py, which reads the files
precomputed by a_prep_reco_data.py. The app never ranks or filters products itself.

Page layout:
    sidebar     : pick a customer and see their shopping style (shopper category)
                  and how many products they bought in the calibration period.
    main column : a chat. Before the first message, three suggestion buttons offer
                  sample queries; afterwards the conversation history is shown.

Streamlit reruns this whole script on every interaction, so anything that must
survive between reruns lives in st.session_state:
    recommendation_agent : the loaded agent (loading reads several large files)
    messages             : chat history for the current customer, as
                           {'role': 'user' | 'assistant', 'content': str, 'error'?: bool}
    active_customer      : customer the history belongs to; changing it clears history
    selected_customer    : the sidebar selectbox value (set by the widget key)

Run from the project folder: streamlit run c_app.py
Set OPENAI_API_KEY (and optionally RECOMMENDATION_MODEL) as for the agent.
"""

from __future__ import annotations

import logging

import streamlit as st


LOGGER = logging.getLogger(__name__)
# Fixed assistant greeting, shown above every conversation but never stored in history.
WELCOME = 'Tell me what you’re looking for, and I’ll help you find a few good options.'
# Shown to the user instead of the exception; the details go to the log.
REQUEST_ERROR = 'I couldn’t complete that request just now. Please try again in a moment.'


def get_agent():
    """Keep the agent in this browser session instead of reloading it on every message.

    The import is deferred so the page can render (and show a spinner) before the
    heavy agent module and its data files load. Any loading error propagates to
    main(), which shows a retry screen.

    Returns:
        The session's RecommendationAgent instance.
    """
    if 'recommendation_agent' not in st.session_state:
        from b_recommendation_agent import RecommendationAgent

        st.session_state.recommendation_agent = RecommendationAgent()
    return st.session_state.recommendation_agent


def reset_conversation():
    """Clear the chat history; also used as the 'New conversation' button callback."""
    st.session_state.messages = []


def render_message(message):
    """Draw one chat message in its role's bubble.

    Args:
        message: dict with 'role' ('user' or 'assistant') and 'content' (markdown text).
            If 'error' is true, the content is shown as an error box instead.
    """
    with st.chat_message(message['role']):
        if message.get('error'):
            st.error(message['content'])
        else:
            st.markdown(message['content'])


def submit_message(agent, customer_id, prompt):
    """Forward the exact request and display only the agent's conversational answer.

    Both the user's message and the reply (or the error notice) are appended to
    st.session_state.messages, so they are redrawn on later reruns.

    Args:
        agent: the RecommendationAgent returned by get_agent().
        customer_id: Customer_ID selected in the sidebar.
        prompt: the user's text, or the query behind a suggestion button.
    """
    user_message = {'role': 'user', 'content': prompt}
    st.session_state.messages.append(user_message)
    # render_message shows user message in the chat interface
    render_message(user_message)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Finding a few good options for you…'):
                result = agent.recommend(
                    customer_id,
                    prompt,
                    conversation_history=[
                        message.copy()
                        for message in st.session_state.messages[:-1]
                        if not message.get('error')
                    ],
                )
                # The agent's state holds intermediate fields too; only the final
                # written answer is meant for the customer.
                answer = result['final_answer']
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError('Agent returned an empty conversational response.')
            st.markdown(answer)
            assistant_message = {'role': 'assistant', 'content': answer}
        except Exception:
            # Never show a traceback in the chat; log it and show a friendly notice.
            LOGGER.exception('Recommendation request failed for customer %s', customer_id)
            st.error(REQUEST_ERROR)
            assistant_message = {'role': 'assistant', 'content': REQUEST_ERROR, 'error': True}
    st.session_state.messages.append(assistant_message)


def main():
    """Build the page: load the agent, draw the sidebar and chat, and handle new input."""
    st.set_page_config(page_title='Customer Concierge', page_icon='✦', layout='centered')
    # Light styling: narrower column, rounded chat bubbles, full-width buttons.
    st.markdown('''
    <style>
      .block-container { max-width: 880px; padding-top: 2.6rem; }
      h1 { letter-spacing: -0.045em; font-weight: 650 !important; }
      [data-testid="stChatMessage"] { border-radius: 16px; padding: 1.15rem; }
      [data-testid="stChatMessage"] p { line-height: 1.65; }
      .stButton > button { width: 100%; border-radius: 10px; }
    </style>
    ''', unsafe_allow_html=True)
    st.caption('✦ CUSTOMER CONCIERGE')
    st.title('A little inspiration for your next shop.')
    st.caption('Ask for familiar favourites, something different, or the latest catalogue additions.')

    # Load the agent and the list of customers. On failure, show a retry button and a
    # disabled chat box, then stop so nothing below runs without an agent.
    try:
        # st.spinner show a temporary message while system get ready
        with st.spinner('Getting your shopping assistant ready…'):
            agent = get_agent()
        customer_ids = sorted(int(customer) for customer in agent.customer_ids.values())
        if not customer_ids:
            raise ValueError('No customers are available.')
    except Exception:
        LOGGER.exception('Shopping assistant could not be initialized')
        st.error('The shopping assistant is unavailable right now. Please try again shortly.')
        if st.button('Try again', key='retry_initialization'):
            st.rerun()
        st.chat_input('The shopping assistant is currently unavailable', disabled=True)
        st.stop()

    # Sidebar: customer picker plus a short profile of the chosen customer.
    with st.sidebar:
        st.subheader('Your shopper')
        customer_id = st.selectbox(
            'Customer', customer_ids,
            format_func=lambda value: f'Customer {value}', key='selected_customer',
            help='Choose a customer to explore their personal recommendations.',
        )
        try:
            # shopper_codes holds the customer category (0/1/2) computed offline.
            shopper_type = agent.SHOPPER_TYPES[int(agent.shopper_codes[customer_id])]
            purchased_products = len(agent.history[customer_id])
        except Exception:
            LOGGER.exception('Customer profile could not be displayed')
            st.error('This customer profile is unavailable. Please choose another customer.')
            st.stop()
        st.caption('SHOPPING STYLE')
        st.subheader(shopper_type.capitalize())
        st.metric('Products previously purchased', purchased_products)
        st.divider()
        st.button('＋ New conversation', key='new_conversation', on_click=reset_conversation)
        st.caption('Switching customers starts a fresh conversation. Your messages stay visible during this session.')

    # A conversation belongs to one customer: start fresh when the selection changes.
    if st.session_state.get('active_customer') != customer_id:
        reset_conversation()
        st.session_state.active_customer = customer_id
    st.session_state.setdefault('messages', [])

    # Greeting first, then either suggestion buttons (empty chat) or the history.
    suggestion = None
    render_message({'role': 'assistant', 'content': WELCOME})
    if not st.session_state.messages:
        # (button label, query sent to the agent); one per intent the agent supports.
        options = [
            ('Something new', 'I want something new under £5.'),
            ('Buy again', 'Recommend a few products I might want to buy again.'),
            ('Latest additions', 'Show me products newly added to the catalogue.'),
        ]
        for column, (label, query) in zip(st.columns(3), options):
            with column:
                if st.button(label, key=f'suggestion_{label}'):
                    suggestion = query
    else:
        for message in st.session_state.messages:
            render_message(message)

    # A clicked suggestion and typed input are handled the same way.
    prompt = st.chat_input('What would you like to find? e.g. “Something new under £5”', key='shopping_message')
    if suggestion or prompt:
        submit_message(agent, customer_id, suggestion or prompt)


if __name__ == '__main__':
    main()
