data collection & processing
- dune query for swap events between major coins
    - hype, eth, btc <> usdt, usdc
    - hype <> eth <> btc
- get token0, token1 for each pool (you have to find pool addresses from factory first)
- calculate pnl (including gas fee) (1m markout; if not enough subscribe tardis and integrate it)
- master table of entire swap records (1 tx = 1 row)
- analyze to answer following questions:
    - in the perspective of searchers
        - market share over time
        - pattern of searchers (top N with 95% market share)

    - in the perspective of pools
        - cumulative pnl of pools over time (with normalization)

발표 흐름
- 팀원 소개
- mev 설명
- 우리 시도 설명
- 도대체 누가 그렇게 많이 해먹나 보자
- 통계 분석한 방법 설명
- 결과 및 인사이트 설명
- 마무리