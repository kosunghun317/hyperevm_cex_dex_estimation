from hyperliquid.info import Info
from hyperliquid.utils import constants
import pprint

info = Info(constants.MAINNET_API_URL, True, None)

spot_meta = info.spot_meta()
spot_meta_and_asset_ctxs = info.spot_meta_and_asset_ctxs()

# get token metadata
for token in spot_meta_and_asset_ctxs[0]["tokens"]:
    if token["name"] in {
        "USDC",
        "USD₮0",
        "USDT",
        "USDT0",
        "UETH",
        "ETH",
        "UBTC",
        "BTC",
        "HYPE",
    }:
        pprint.pprint(
            [
                token["name"],
                token["index"],
                token["evmContract"],
            ]
        )

# get pair metadata
for meta in spot_meta["universe"]:
    if (
        (meta["tokens"] == [150, 0])
        or (meta["tokens"] == [197, 0])
        or (meta["tokens"] == [221, 0])
        or (meta["tokens"] == [268, 0])
    ):
        pprint.pprint(meta)
