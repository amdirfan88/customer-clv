"""Conversational recommendations backed by the saved rank matrices.

Run: python recommendation_agent.py --customer-id 12347 --query "Something new under £5"
Set OPENAI_API_KEY and optionally RECOMMENDATION_MODEL in the environment or .env.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomerRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    intent: Literal['repeat', 'new', 'new_catalog', 'automatic'] = 'automatic'
    catalog_scope: Literal['all_available', 'newly_added'] = 'all_available'
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    min_price_inclusive: bool = True
    max_price_inclusive: bool = True
    count: int = Field(default=5, ge=1, le=20)
    search_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    exclude_stockcodes: list[str] = Field(default_factory=list)
    clarification: str | None = None

    @model_validator(mode='after')
    def valid_price_range(self):
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError('Minimum price must not exceed maximum price.')
        return self


class ProductReason(BaseModel):
    stockcode: str
    reason: str


class ConversationResponse(BaseModel):
    introduction: str
    reasons: list[ProductReason] = Field(default_factory=list)


class RecommendationState(TypedDict, total=False):
    customer_id: int
    user_query: str
    request: dict
    shopper_type: str
    shopper_code: int
    purchase_history: list[str]
    intent: str
    ranking_model: str
    recommendations: list[dict]
    final_answer: str


class RecommendationAgent:
    """LLM understanding and conversation; deterministic ranking and constraints."""

    SHOPPER_TYPES = {0: 'infrequent shopper', 1: 'repeat shopper', 2: 'product explorer'}

    def __init__(self, data_dir: str | Path | None = None, llm=None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'data'
        working = self.data_dir / 'working_data'
        given = self.data_dir / 'given_data'
        with (working / 'recommendation_index_maps.pkl').open('rb') as file:
            maps = pickle.load(file)
        self.customer_ids = maps['index_to_customer_id']
        self.product_codes = maps['index_to_stockcode']
        if set(self.customer_ids) != set(range(len(self.customer_ids))):
            raise ValueError('Customer indices must be contiguous from zero.')
        if set(self.product_codes) != set(range(len(self.product_codes))):
            raise ValueError('Product indices must be contiguous from zero.')
        self.customer_rows = {int(code): index for index, code in self.customer_ids.items()}
        if len(self.customer_rows) != len(self.customer_ids):
            raise ValueError('Customer index map contains duplicate IDs.')
        self.ordered_products = [str(self.product_codes[index]) for index in range(len(self.product_codes))]
        if len(set(self.ordered_products)) != len(self.ordered_products):
            raise ValueError('Product index map contains duplicate codes.')
        self.catalog = pd.read_csv(
            given / 'prod_catalog_holdout.csv', dtype={'StockCode': 'string'},
        ).set_index('StockCode')
        if not self.catalog.index.is_unique:
            raise ValueError('Holdout catalogue contains duplicate stock codes.')
        if not set(self.ordered_products).issubset(self.catalog.index):
            raise ValueError('Product index map does not match the holdout catalogue.')
        self.catalog = self.catalog.loc[self.ordered_products]
        calibration_products = set(pd.read_csv(
            given / 'prod_catalog_calib.csv', usecols=['StockCode'],
            dtype={'StockCode': 'string'},
        )['StockCode'].dropna())
        self.new_catalog_products = set(self.ordered_products) - calibration_products
        categories = pd.read_csv(working / 'customer_category.csv').set_index('Customer_ID')
        if not categories.index.is_unique:
            raise ValueError('Customer category file contains duplicate IDs.')
        self.shopper_codes = categories['customer_category'].to_dict()
        history = pd.read_csv(
            working / 'calib_all_transaction.csv', usecols=['Customer_ID', 'StockCode'],
            dtype={'Customer_ID': 'Int64', 'StockCode': 'string'},
        ).dropna()
        self.history = history.groupby('Customer_ID')['StockCode'].agg(set).to_dict()
        if set(self.customer_rows) != set(self.history):
            raise ValueError('Customer index map does not match cleaned purchase history.')
        if any(self.shopper_codes.get(customer) not in self.SHOPPER_TYPES for customer in self.customer_rows):
            raise ValueError('Customer categories are missing or invalid; regenerate customer_category.csv.')
        self.rank_paths = {
            'two_tower': working / 'two_tower_ranks.parquet',
            'repeat_purchase': working / 'repeat_purchase_ranks.parquet',
        }
        self.rank_matrices: dict[str, np.ndarray] = {}

        if llm is None:
            load_dotenv(Path(__file__).resolve().parent / '.env')
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError('Set OPENAI_API_KEY in the environment or project .env file.')
            llm = ChatOpenAI(
                model=os.getenv('RECOMMENDATION_MODEL', 'gpt-5-mini'),
                timeout=60, max_retries=2,
            )
        self.request_llm = llm.with_structured_output(CustomerRequest)
        self.response_llm = llm.with_structured_output(ConversationResponse)

        builder = StateGraph(RecommendationState)
        builder.add_node('understand_query', self.understand_query)
        builder.add_node('determine_intent', self.determine_intent)
        builder.add_node('determine_shopper_type', self.determine_shopper_type)
        builder.add_node('select_ranking', self.select_ranking)
        builder.add_node('apply_constraints', self.apply_constraints)
        builder.add_node('retrieve_products', self.retrieve_products)
        builder.add_node('respond', self.respond)
        builder.add_edge(START, 'understand_query')
        builder.add_edge('understand_query', 'determine_intent')
        builder.add_edge('determine_intent', 'determine_shopper_type')
        builder.add_edge('determine_shopper_type', 'select_ranking')
        builder.add_edge('select_ranking', 'apply_constraints')
        builder.add_edge('apply_constraints', 'retrieve_products')
        builder.add_edge('retrieve_products', 'respond')
        builder.add_edge('respond', END)
        self.graph = builder.compile()

    def understand_query(self, state: RecommendationState):
        response = self.request_llm.invoke([
            SystemMessage(content=(
                'Extract the shopping request; never invent customer history or products. '
                'Intent repeat means buy again; new means products never bought by this customer; '
                'new_catalog means explicitly newly introduced catalogue items. Use automatic when '
                'there is no preference, letting shopper type decide. The catalogue is the available '
                'holdout catalogue: merely asking to browse this catalogue does not mean newly added. '
                'Set catalog_scope=newly_added only for explicit newly added/latest arrivals requests. '
                'Extract price bounds and count (default 5, maximum 20); under/below is exclusive, '
                'at most/up to is inclusive; over/above a minimum is exclusive. '
                'Extract literal description search terms and exclusions; '
                'do not include generic words such as recommend or products. Prices are historical '
                'average unit prices in GBP. Do not convert currencies. For a different currency or '
                'unsupported constraints (such as stock availability, live price, shipping, size '
                'measurements or a total basket budget), set clarification to an honest question '
                'rather than pretending those constraints can be checked.'
            )),
            HumanMessage(content=state['user_query']),
        ])
        request = response if isinstance(response, CustomerRequest) else CustomerRequest.model_validate(response)
        return {'request': request.model_dump()}

    def determine_intent(self, state: RecommendationState):
        request = state['request']
        intent = request['intent']
        if intent == 'automatic' and request['catalog_scope'] == 'newly_added':
            intent = 'new_catalog'
        return {'intent': intent}

    def determine_shopper_type(self, state: RecommendationState):
        customer = state['customer_id']
        code = int(self.shopper_codes[customer])
        return {'shopper_code': code, 'shopper_type': self.SHOPPER_TYPES[code],
                'purchase_history': sorted(self.history[customer])}

    def select_ranking(self, state: RecommendationState):
        intent = state['intent']
        if intent == 'automatic':
            if state['shopper_code'] == 1:
                intent = 'repeat'
            elif state['shopper_code'] == 2:
                intent = 'new'
        model = 'repeat_purchase' if intent == 'repeat' else 'two_tower'
        return {'intent': intent, 'ranking_model': model}

    def apply_constraints(self, state: RecommendationState):
        if state['request']['clarification']:
            return {'recommendations': []}
        model = state['ranking_model']
        if model not in self.rank_matrices:
            table = pd.read_parquet(self.rank_paths[model])
            expected = (len(self.customer_ids), len(self.product_codes))
            if table.shape != expected or list(table.columns) != [str(i) for i in range(expected[1])]:
                raise ValueError(f'{self.rank_paths[model].name} does not match the index dictionaries.')
            self.rank_matrices[model] = table.to_numpy()
        ranks = self.rank_matrices[model][self.customer_rows[state['customer_id']]]
        number_of_products = len(self.product_codes)
        mask = (ranks > 0) & (ranks < number_of_products)
        if model == 'two_tower':
            mask &= ranks <= 1000
        history = set(state['purchase_history'])
        codes = np.array(self.ordered_products)
        if state['intent'] == 'repeat':
            mask &= np.isin(codes, list(history))
        elif state['intent'] in {'new', 'new_catalog'}:
            mask &= ~np.isin(codes, list(history))
        request = state['request']
        if state['intent'] == 'new_catalog' or request['catalog_scope'] == 'newly_added':
            mask &= np.isin(codes, list(self.new_catalog_products))
        prices = pd.to_numeric(self.catalog['avgprice'], errors='coerce').to_numpy()
        mask &= np.isfinite(prices) & (prices > 0)
        if request['min_price'] is not None:
            mask &= prices >= request['min_price'] if request['min_price_inclusive'] else prices > request['min_price']
        if request['max_price'] is not None:
            mask &= prices <= request['max_price'] if request['max_price_inclusive'] else prices < request['max_price']
        descriptions = self.catalog['Description'].fillna('').astype(str)
        for term in request['search_terms']:
            if term.strip():
                mask &= descriptions.str.contains(term.strip(), case=False, regex=False).to_numpy()
        for term in request['exclude_terms']:
            if term.strip():
                mask &= ~descriptions.str.contains(term.strip(), case=False, regex=False).to_numpy()
        mask &= ~np.isin(codes, request['exclude_stockcodes'])
        indices = np.flatnonzero(mask)
        selected = indices[np.argsort(ranks[indices], kind='stable')][:request['count']]
        return {'recommendations': [{'stockcode': self.ordered_products[i], 'rank': int(ranks[i])} for i in selected]}

    def retrieve_products(self, state: RecommendationState):
        products = []
        for recommendation in state['recommendations']:
            code = recommendation['stockcode']
            row = self.catalog.loc[code]
            products.append({
                **recommendation,
                'description': str(row['Description']),
                'historical_average_price': float(row['avgprice']),
                'previously_purchased': code in self.history[state['customer_id']],
                'newly_added_to_catalogue': code in self.new_catalog_products,
            })
        return {'recommendations': products}

    def respond(self, state: RecommendationState):
        if state['request']['clarification']:
            return {'final_answer': state['request']['clarification']}
        products = state['recommendations']
        if not products:
            return {'final_answer': (
                'I could not find recommendations matching those preferences and price limits '
                'in the available rankings. Would you like to broaden the request?'
            )}
        response = self.response_llm.invoke([
            SystemMessage(content=(
                'Write a brief, friendly introduction and one reason for each supplied product. '
                'Use only the provided facts and customer request. Never add products, prices, '
                'availability, discounts, or unsupported product properties. Do not mention '
                'model names, ranks, shopper categories, or implementation details. '
                'Avoid repeating product names or prices in the introduction/reasons; they are '
                'rendered separately. Return exactly the supplied stockcodes, each once.'
            )),
            HumanMessage(content=json.dumps({
                'request': state['user_query'], 'intent': state['intent'],
                'products': products,
            })),
        ])
        response = response if isinstance(response, ConversationResponse) else ConversationResponse.model_validate(response)
        reasons = {item.stockcode: item.reason for item in response.reasons}
        if len(response.reasons) != len(products) or set(reasons) != {p['stockcode'] for p in products}:
            raise ValueError('The response model returned product references outside the recommendation results.')
        lines = [response.introduction]
        for product in products:
            lines.append(f"- {product['description']} ({product['stockcode']}) — "
                         f"£{product['historical_average_price']:.2f}: {reasons[product['stockcode']]}")
        lines.append('Prices shown are historical average unit prices, not live prices.')
        return {'final_answer': '\n'.join(lines)}

    def recommend(self, customer_id: int, user_query: str) -> RecommendationState:
        customer_id = int(customer_id)
        if customer_id not in self.customer_rows:
            raise ValueError(f'Customer {customer_id} has no cleaned purchase history in the saved rankings.')
        if not user_query.strip():
            raise ValueError('Please provide a shopping request.')
        return self.graph.invoke({'customer_id': customer_id, 'user_query': user_query.strip()})


def main():
    parser = argparse.ArgumentParser(description='Personalized recommendations using saved rank matrices.')
    parser.add_argument('--customer-id', type=int, required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--data-dir', type=Path, default=None)
    args = parser.parse_args()
    agent = RecommendationAgent(data_dir=args.data_dir)
    print(agent.recommend(args.customer_id, args.query)['final_answer'])


if __name__ == '__main__':
    main()
