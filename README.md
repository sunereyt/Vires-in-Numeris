# Vires-in-Numeris

Strategy for [Freqtrade](https://www.freqtrade.io/en/stable/).

Work in progress, you might loose money.

Backtest with protections enabled, eg:

`sudo docker-compose run freqtrade backtesting --strategy ViresInNumeris --config user_data/btconfig.json --timerange 20210601- --max-open-trades 10 --enable-protections`
