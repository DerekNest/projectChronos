import asyncio, aiohttp, json
async def test():
    async with aiohttp.ClientSession() as s:
        async with s.get('https://gamma-api.polymarket.com/markets?closed=true&limit=5&end_date_min=2025-03-01') as r:
            data = await r.json(content_type=None)
            for m in data[:2]:
                tokens = json.loads(m.get('clobTokenIds','[]'))
                print(m.get('question','')[:60])
                print('  endDate:', m.get('endDate'))
                if tokens:
                    url = f'https://clob.polymarket.com/prices-history?market={tokens[0]}&interval=max&fidelity=60'
                    async with s.get(url) as r2:
                        h = await r2.json(content_type=None)
                        pts = len(h.get('history', []))
                        print(f'  history points: {pts}')
asyncio.run(test())
