import logging
import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
import time
import pandas_ta as pta
from collections import Counter

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade

import locale
locale.setlocale(category=locale.LC_ALL, locale='')

log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 10,
    }

    f_buys = './user_data/vinbuys.txt'
    f_trades = './user_data/vintrades.txt'
    write_to_csv = False
    df_csv = './user_data/df.csv'

    stoploss = -0.10

    # Trailing stoploss (not used)
    trailing_stop = False
    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    info_timeframe_1h = '1h'
    info_timeframe_1d = '1d'

    # Backtest Age Filter emulation
    has_bt_agefilter = True
    bt_min_age_days = 21

    # Exchange Downtime protection
    has_downtime_protection = False

    # Minimal volume filters
    min_vol_candle = 1000
    min_vol_1h = 10000

    # Maximum number of concurrent buy signals (0 is disable)
    max_concurrent_buy_signals = 14
    # Maximum number of buys with the same buy tag (0 is disable)
    max_same_buy_tags = 10
    # Create custom dictionary
    custom_buy_info = {}

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 144

    # # Optional order type mapping.
    # order_types = {
    #     'buy': 'limit',
    #     'sell': 'limit',
    #     'trailing_stop_loss': 'limit',
    #     'stoploss': 'limit',
    #     'stoploss_on_exchange': False,
    #     'stoploss_on_exchange_interval': 60,
    #     'stoploss_on_exchange_limit_ratio': 0.99
    # }


    def informative_1d_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1d: DataFrame = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1d)

        if self.config['runmode'].value not in ('live', 'dry_run'):
            if self.has_bt_agefilter:
                informative_1d['bt_agefilter'] = informative_1d['volume'].rolling(window=self.bt_min_age_days, min_periods=self.bt_min_age_days).count()

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1d_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1d

    def informative_1h_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h: DataFrame = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1h)

        # Chopiness
        informative_1h['chop_84'] = qtpylib.chopiness(df, 84)

        # CCI Oscillator
        cci_84: Series = ta.CCI(df, timeperiod=84)
        cci_84_max: Series = cci_84.rolling(self.startup_candle_count).max()
        cci_84_min: Series = cci_84.rolling(self.startup_candle_count).min()
        informative_1h['cci_osc_84'] = (cci_84 / cci_84_max).where(cci_84 > 0, -cci_84 / cci_84_min)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1h

    def normal_tf_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()

        # MFI
        df['mfi_14'] = ta.MFI(df, timeperiod=14)
        df['mfi_42'] = ta.MFI(df, timeperiod=42)

        # CMF
        df['cmf_3'] = chaikin_money_flow(df, 3)
        df['cmf_14'] = chaikin_money_flow(df, 14)
        df['cmf_42'] = chaikin_money_flow(df, 42)

        # RSI
        df['rsi_3'] = ta.RSI(df, timeperiod=3)
        df['rsi_14'] = ta.RSI(df, timeperiod=14)
        df['rsi_42'] = ta.RSI(df, timeperiod=42)

        # Chopiness
        df['chop_42']= qtpylib.chopiness(df, 42)

        # CRSI
        crsi_closechange = df['close'] / df['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        df['crsi_100'] = (ta.RSI(df['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(df['close'], 100)) / 3
        df['crsi_480'] = (ta.RSI(df['close'], timeperiod=6) + ta.RSI(crsi_updown, timeperiod=3) + ta.ROC(df['close'], 480)) / 3

        # CTI
        df['cti_3'] = pta.cti(df['close'], length=3)
        df['cti_14'] = pta.cti(df['close'], length=14)
        df['cti_42'] = pta.cti(df['close'], length=42)

        # CCI Oscillator
        cci_42: Series = ta.CCI(df, timeperiod=42)
        cci_42_max: Series = cci_42.rolling(self.startup_candle_count).max()
        cci_42_min: Series = cci_42.rolling(self.startup_candle_count).min()
        df['cci_osc_42'] = (cci_42 / cci_42_max).where(cci_42 > 0, -cci_42 / cci_42_min)

        # Momentum
        mom_14 = ta.MOM(df, timeperiod=14)
        df['mom_14'] = mom_14
        upperband, middleband, lowerband = ta.BBANDS(mom_14, timeperiod=42, nbdevup=2.0, nbdevdn=2.0, matype=0) # 0 = sma, 1 = ema, 2 = wma
        df['mom_14_upp'] = upperband
        df['mom_14_mid'] = middleband
        df['mom_14_low'] = lowerband

        # Volume
        df['volume_12'] = df['volume'].rolling(12).sum()

        # Maximum positive and negative change in one hour
        df['pump'] = df[['open', 'close']].max(axis=1).rolling(window=12, min_periods=0).max() / df[['open', 'close']].min(axis=1).shift(1).rolling(window=12, min_periods=0).min()
        df['dump'] = df[['open', 'close']].min(axis=1).rolling(window=12, min_periods=0).min() / df[['open', 'close']].max(axis=1).shift(1).rolling(window=12, min_periods=0).max()

        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                df['live_data_ok'] = (df['volume'].rolling(window=72, min_periods=72).min() > 0)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] normal_tf_indicators took: {tok - tik:0.4f} seconds.")

        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        if self.info_timeframe_1d != 'none':
            informative_1d = self.informative_1d_indicators(df, metadata)
            df = merge_informative_pair(df, informative_1d, self.timeframe, self.info_timeframe_1d, ffill=True)
            drop_columns = [f"{s}_{self.info_timeframe_1d}" for s in ['date','open', 'high', 'low', 'close', 'volume']]
            df.drop(columns=df.columns.intersection(drop_columns), inplace=True)

        if self.info_timeframe_1h != 'none':
            informative_1h = self.informative_1h_indicators(df, metadata)
            df = merge_informative_pair(df, informative_1h, self.timeframe, self.info_timeframe_1h, ffill=True)
            drop_columns = [f"{s}_{self.info_timeframe_1h}" for s in ['date']]
            df.drop(columns=df.columns.intersection(drop_columns), inplace=True)

        df = self.normal_tf_indicators(df, metadata)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] Populate indicators took a total of: {tok - tik:0.4f} seconds.")

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        df.loc[:, 'buy_tag'] = ''

        conditions.append(df['volume'] >= self.min_vol_candle)
        conditions.append(df['volume_12'] >= self.min_vol_1h)
        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                conditions.append(df['live_data_ok'])
        else:
            if self.has_bt_agefilter:
                conditions.append(df['bt_agefilter_1d'] >= self.bt_min_age_days)
        logic = []
        logic.append(df['mom_14'] < df['mom_14_low'])
        buy = reduce(lambda x, y: x & y, logic)
        df.loc[buy, 'buy_tag'] +='mom'
        logic = []
        logic.append(df['rsi_14'] < 35)
        buy = reduce(lambda x, y: x & y, logic)
        df.loc[buy, 'buy_tag'] +='rsi'
        logic = []
        logic.append(df['mfi_14'] < 25)
        buy = reduce(lambda x, y: x & y, logic)
        df.loc[buy, 'buy_tag'] +='mfi'
        logic = []
        logic.append(df['cti_14'] < -0.75)
        logic.append(df['cti_14'] >= df['cti_14'].shift(1))
        buy = reduce(lambda x, y: x & y, logic)
        df.loc[buy, 'buy_tag'] +='cti'
        logic = []
        logic.append(df['cmf_14'] > -0.05)
        logic.append(df['cmf_14'] >= df['cmf_14'].shift(1))
        buy = reduce(lambda x, y: x & y, logic)
        df.loc[buy, 'buy_tag'] +='cmf'

        conditions.append(df.loc[:, 'buy_tag'].str.len() >= 6)
        if conditions:
            df.loc[:, 'buy'] = reduce(lambda x, y: x & y, conditions)

            df_buy: DataFrame = df.loc[df.loc[:, 'buy'], ['date', 'buy_tag', 'buy']]
            for index, row in df_buy.iterrows():
                buy_date = row['date']
                if buy_date not in self.custom_buy_info:
                    self.custom_buy_info[buy_date] = {}
                    self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_tags'] = row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_signals'] = 1
                else:
                    self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_tags'] += row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_signals'] += 1

        if self.config['runmode'].value not in ('live', 'dry_run'):
            if self.write_to_csv:
                df['pair'] = metadata['pair']
                with open(self.df_csv, 'a') as f:
                    df.to_csv(f, sep='\t', header=f.tell()==0, index=False)

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        df.loc[:, 'sell'] = False
        df.loc[:, 'sell_tag'] = ''

        logic = []
        logic.append(df['mom_14'] > df['mom_14_upp'])
        sell = reduce(lambda x, y: x & y, logic)
        df.loc[sell, 'sell_tag'] +='mom'
        logic = []
        logic.append(df['rsi_14'] > 65)
        sell = reduce(lambda x, y: x & y, logic)
        df.loc[sell, 'sell_tag'] +='rsi'
        logic = []
        logic.append(df['mfi_14'] > 75)
        sell = reduce(lambda x, y: x & y, logic)
        df.loc[sell, 'sell_tag'] +='mfi'
        logic = []
        logic.append(df['cti_14'] > 0.75)
        logic.append(df['cti_14'] <= df['cti_14'].shift(1))
        sell = reduce(lambda x, y: x & y, logic)
        df.loc[sell, 'sell_tag'] +='cti'
        logic = []
        logic.append(df['cmf_14'] < 0.05)
        logic.append(df['cmf_14'] <= df['cmf_14'].shift(1))
        sell = reduce(lambda x, y: x & y, logic)
        df.loc[sell, 'sell_tag'] +='cmf'

        conditions.append(df['sell_tag'].str.len() >= 6)
        if conditions:
            df.loc[:, 'sell'] = reduce(lambda x, y: x & y, conditions)

        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df[df['date'] >= trade_open_date]
        if df_trade.empty:
            return None

        candle_1 = df.iloc[-1]
        max_close_candle = df_trade.nlargest(1, columns=['close'])
        min_close_candle = df_trade.nsmallest(1, columns=['close'])
        max_close = max_close_candle['close'].iloc[0]
        min_close = min_close_candle['close'].iloc[0]

        # use close instead of trade prices
        current_rate = candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        trade.max_rate = max_close
        trade.min_rate = min_close

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        # buy_tags = buy_tag.split()

        # do nothing with small profits
        if len(candle_1['sell_tag']) >= 6 and current_profit > 0.01:
                return f"{candle_1['sell_tag']} ( {buy_tag})"

        return None

    def bot_loop_start(self, **kwargs) -> None:
        if self.config['runmode'].value not in ('live', 'dry_run'):
            with open(self.f_buys, 'w') as f:
                print('pair;date open;trade open rate;buy tags;close price;sma_75;ema_25;ewo_25;ewo;rsi_4;rsi_20;r_14;cmf_14;cti_14;rsi_14;mfi_14;cmf_42;cti_42;rsi_42;mfi_42;rsi_14_1h;crsi_100;crsi_480;crsi_1h;crsi_480_1h;cci_osc_42;cci_osc_84_1h;pump;dump;chop_84_1h;ema_vwma_osc_32;ema_vwma_osc_64;ema_vwma_osc_96;cmf_14;cti_14;rsi_14;mfi_14;cmf_42;cti_42;rsi_42;mfi_42;rsi_14_1h;crsi_100;crsi_480;crsi_1h;crsi_480_1h;cci_osc_42;cci_osc_84_1h', file=f)
            with open(self.f_trades, 'w') as f:
                print('pair;date open;trade open rate;date close;trade rate;buy tags;sell reason;profit;max profit;max loss;max rate;min rate;max close date;min close date', file=f)
            if self.write_to_csv:
                with open(self.df_csv, 'w') as f:
                    pass

        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # candle_1: Series = df.iloc[-1]
        # candle_2: Series = df.iloc[-2]
        # buy_candle_date = candle_1['date']

        # if buy_candle_date in self.custom_buy_info.keys():
        #     buy_tags = self.custom_buy_info[buy_candle_date][pair]
        #     # do not buy when there are many buy signals and concurrent buy tags
        #     if  self.max_concurrent_buy_signals > 0 and self.max_same_buy_tags > 0:
        #         buy_info = self.custom_buy_info[buy_candle_date]
        #         buy_signal_count = buy_info['buy_signals']
        #         buy_tag, buy_tag_count = Counter(buy_info['buy_tags'].split()).most_common()[0]
        #         if buy_signal_count > self.max_concurrent_buy_signals and buy_tag_count > self.max_same_buy_tags:
        #             log.info(f"Buy for pair {pair} with buy tag {buy_tags}on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {self.max_concurrent_buy_signals}) and buy tag {buy_tag} was triggered {buy_tag_count} times (max = {self.max_same_buy_tags}).")
        #             return False

        #     if self.config['runmode'].value not in ('live', 'dry_run'):
        #         close_price = candle_1['close']
        #         indicator = []
        #         indicator.append(candle_1['cmf_14'])
        #         indicator.append(candle_1['cti_14'])
        #         indicator.append(candle_1['rsi_14'])
        #         indicator.append(candle_1['mfi_14'])
        #         indicator.append(candle_1['cmf_42'])
        #         indicator.append(candle_1['cti_42'])
        #         indicator.append(candle_1['rsi_42'])
        #         indicator.append(candle_1['mfi_42'])
        #         indicator.append(candle_1['pump'])
        #         indicator.append(candle_1['dump'])
        #         indicator.append(candle_1['chop_84_1h'])
        #         indicator.append(candle_2['cmf_14'])
        #         indicator.append(candle_2['cti_14'])
        #         indicator.append(candle_2['rsi_14'])
        #         indicator.append(candle_2['mfi_14'])
        #         indicator.append(candle_2['cmf_42'])
        #         indicator.append(candle_2['cti_42'])
        #         indicator.append(candle_2['rsi_42'])
        #         indicator.append(candle_2['mfi_42'])
        #         with open(self.f_buys, 'a') as f:
        #             print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};{close_price:.10n}", *indicator, sep=';', file=f)
        # else:
        #     log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")

        return True

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # do not sell existing positions when there are many buy signals or concurrent buy tags
        # if  self.max_concurrent_buy_signals > 0 or self.max_same_buy_tags > 0:
        #     candle_1: Series = df.iloc[-1]
        #     candle_date = candle_1['date']
        #     if candle_date in self.custom_buy_info.keys():
        #         buy_info = self.custom_buy_info[candle_date]
        #         if self.max_concurrent_buy_signals > 0:
        #             buy_signal_count = buy_info['buy_signals']
        #             if buy_signal_count > self.max_concurrent_buy_signals:
        #                 log.info(f"Sell for pair {pair} on candle {candle_date} with sell reason {sell_reason} is cancelled. There are {buy_signal_count} concurrent buy signals which is higher than the maximum ({self.max_concurrent_buy_signals}).")
        #         if self.max_same_buy_tags > 0:
        #             buy_tag, buy_tag_count = Counter(buy_info['buy_tags'].split()).most_common()[0]
        #             if buy_tag_count > self.max_same_buy_tags:
        #                 log.info(f"Sell for pair {pair} on candle {candle_date} with sell reason {sell_reason} is cancelled. Buy tag {buy_tag} was triggered {buy_tag_count} times which is higher than the maximum ({self.max_same_buy_tags}).")
        #         if buy_signal_count > self.max_concurrent_buy_signals or buy_tag_count > self.max_same_buy_tags:
        #             return False

        # if self.config['runmode'].value not in ('live', 'dry_run'):
        #     trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        #     trade_close_date = timeframe_to_prev_date(self.timeframe, trade.close_date_utc)
        #     buy_tag = trade.buy_tag if trade is not None else 'empty'

        #     df_trade = df[(df['date'] >= trade_open_date) & (df['date'] <= trade_close_date)]
        #     if df_trade.empty:
        #         log.warning(f"confirm_trade_exit: Empty trade df for pair {pair} on trade open date {trade_open_date}.")
        #         return False

        #     max_close_candle = df_trade.nlargest(1, columns=['close'])
        #     min_close_candle = df_trade.nsmallest(1, columns=['close'])
        #     min_close_date = min_close_candle['date'].to_numpy()[0]
        #     max_close_date = max_close_candle['date'].to_numpy()[0]
        #     profit = (rate - trade.open_rate) / trade.open_rate
        #     max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
        #     max_loss = (trade.min_rate - trade.open_rate) / trade.open_rate

        #     with open(self.f_trades, 'a') as f:
        #         print(f'{pair};{trade_open_date};{trade.open_rate:.10n};{trade_close_date};{rate:.10n};{buy_tag};{sell_reason.partition(" (")[0]};{profit:.10n};{max_profit:.10n};{max_loss:.10n};{trade.max_rate:.10n};{trade.min_rate:.10n};{max_close_date};{min_close_date};', file=f)

        return True

# Chaikin Money Flow
def chaikin_money_flow(df: DataFrame, n=20, fillna=False) -> Series:
    mfv: Series = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)

    return cmf
