from tardis_dev import get_exchange_details, datasets
import pprint
import os
from dotenv import load_dotenv

load_dotenv()

# pprint.pprint(get_exchange_details("hyperliquid"))

pprint.pprint(
    datasets.download(
        exchange="hyperliquid",
        data_types=["quotes"],
        from_date="2025-08-01T00:00:00.000Z",
        to_date="2025-08-30T23:59:59.999Z",
        symbols=[
            "@107",  # HYPE-USDC
            "@142",  # UBTC-USDC
            "@151",  # UETH-USDC
            "@166",  # USDT0-USDC
        ],
        api_key=os.getenv("TARDIS_API_KEY"),
    )
)
