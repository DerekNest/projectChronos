import asyncio, aiohttp, json

GOLDSKY = 'https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn'

async def test():
    # use the Manchester City YES token from earlier test
    # first get a recent market's token id from gamma
    async with aiohttp.ClientSession() as s:
        async with s.get('https://gamma-api.polymarket.com/markets?closed=true&limit=3&end_date_min=2025-03-01') as r:
            markets = await r.json(content_type=None)
            for m in markets[:2]:
                tokens = json.loads(m.get('clobTokenIds', '[]'))
                if not tokens:
                    continue
                yes_token = tokens[0]
                print(f"market: {m.get('question','')[:50]}")
                print(f"yes_token: {yes_token[:20]}...")

                query = '''{ orderFilledEvents(
                    first: 10
                    orderBy: timestamp
                    orderDirection: asc
                    where: { makerAssetId: "''' + yes_token + '''" }
                ) { timestamp makerAmountFilled takerAmountFilled } }'''

                async with s.post(GOLDSKY, json={'query': query}, headers={'Content-Type': 'application/json'}) as r2:
                    data = await r2.json(content_type=None)
                    events = data.get('data', {}).get('orderFilledEvents', [])
                    print(f"trades (as maker): {len(events)}")

                query2 = '''{ orderFilledEvents(
                    first: 10
                    orderBy: timestamp
                    orderDirection: asc
                    where: { takerAssetId: "''' + yes_token + '''" }
                ) { timestamp makerAmountFilled takerAmountFilled } }'''

                async with s.post(GOLDSKY, json={'query': query2}, headers={'Content-Type': 'application/json'}) as r3:
                    data2 = await r3.json(content_type=None)
                    events2 = data2.get('data', {}).get('orderFilledEvents', [])
                    print(f"trades (as taker): {len(events2)}")
                print()

asyncio.run(test())
