"""Streamlit presentation layer; all recommendations come from the agent.

Run from the project folder: streamlit run c_app.py
"""

from __future__ import annotations

import logging

import streamlit as st


LOGGER = logging.getLogger(__name__)
WELCOME = 'Tell me what you’re looking for, and I’ll help you find a few good options.'
REQUEST_ERROR = 'I couldn’t complete that request just now. Please try again in a moment.'


def get_agent():
    """Keep the agent in this browser session instead of reloading it on every message."""
    if 'recommendation_agent' not in st.session_state:
        from b_recommendation_agent import RecommendationAgent

        st.session_state.recommendation_agent = RecommendationAgent()
    return st.session_state.recommendation_agent


def reset_conversation():
    st.session_state.messages = []


def render_message(message):
    with st.chat_message(message['role']):
        if message.get('error'):
            st.error(message['content'])
        else:
            st.markdown(message['content'])


def submit_message(agent, customer_id, prompt):
    """Forward the exact request and display only the agent's conversational answer."""
    user_message = {'role': 'user', 'content': prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Finding a few good options for you…'):
                result = agent.recommend(customer_id, prompt)
                answer = result['final_answer']
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError('Agent returned an empty conversational response.')
            st.markdown(answer)
            assistant_message = {'role': 'assistant', 'content': answer}
        except Exception:
            LOGGER.exception('Recommendation request failed for customer %s', customer_id)
            st.error(REQUEST_ERROR)
            assistant_message = {'role': 'assistant', 'content': REQUEST_ERROR, 'error': True}
    st.session_state.messages.append(assistant_message)


def main():
    st.set_page_config(page_title='Customer Concierge', page_icon='✦', layout='centered')
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

    try:
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

    with st.sidebar:
        st.subheader('Your shopper')
        customer_id = st.selectbox(
            'Customer', customer_ids,
            format_func=lambda value: f'Customer {value}', key='selected_customer',
            help='Choose a customer to explore their personal recommendations.',
        )
        try:
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

    if st.session_state.get('active_customer') != customer_id:
        reset_conversation()
        st.session_state.active_customer = customer_id
    st.session_state.setdefault('messages', [])

    suggestion = None
    render_message({'role': 'assistant', 'content': WELCOME})
    if not st.session_state.messages:
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

    prompt = st.chat_input('What would you like to find? e.g. “Something new under £5”', key='shopping_message')
    if suggestion or prompt:
        submit_message(agent, customer_id, suggestion or prompt)


if __name__ == '__main__':
    main()
