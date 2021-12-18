# Vires-in-Numeris

Strategy for [Freqtrade](https://www.freqtrade.io/en/stable/).

Work in progress, you might loose money.

I use about 1/12 of the number of pairs for the amount of slots. So a pairlist with 120 pairs means 10 slots for trading.

Backtest with protections enabled, eg:

`sudo docker-compose run freqtrade backtesting --strategy ViresInNumeris --config user_data/120pairsconfig.json --timerange 20210601- --max-open-trades 10 --enable-protections`
