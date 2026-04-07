import asyncio, aiohttp, json

GOLDSKY_URL = 'https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/polymarket-orderbook-resync/prod/gn'

async def test():
    query = '''
    {
      orderFilledEvents(
        first: 20
        orderBy: timestamp
        orderDirection: desc
        where: {
          market: "0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917"
        }
      ) {
        timestamp
        price
        side
        makerAmountFilled
        takerAmountFilled
      }
    }
    '''
    async with aiohttp.ClientSession() as s:
        async with s.post(GOLDSKY_URL, json={'query': query}) as r:
            data = await r.json(content_type=None)
            events = data.get('data', {}).get('orderFilledEvents', [])
            print(f'trades found: {len(events)}')
            if events:
                print('sample:', events[0])

asyncio.run(test())
