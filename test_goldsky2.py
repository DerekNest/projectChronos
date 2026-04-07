import asyncio, aiohttp, json

GOLDSKY = 'https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn'

async def test():
    query = '{ orderFilledEvents(first: 5 orderBy: timestamp orderDirection: desc) { timestamp makerAmountFilled takerAmountFilled makerAssetId } }'
    async with aiohttp.ClientSession() as s:
        async with s.post(GOLDSKY, json={'query': query}, headers={'Content-Type': 'application/json'}) as r:
            data = await r.json(content_type=None)
            events = data.get('data', {}).get('orderFilledEvents', [])
            print(f'events found: {len(events)}')
            if events:
                print('sample:', json.dumps(events[0], indent=2))
            else:
                print('raw response:', json.dumps(data, indent=2)[:500])

asyncio.run(test())
