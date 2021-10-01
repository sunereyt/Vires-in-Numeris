import logging
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from freqtrade.exchange import timeframe_to_prev_date
from pandas import DataFrame, Series
from functools import reduce
import math
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.util import resample_to_interval, resampled_merge
# from technical.indicators import ichimoku
import time
import pandas_ta as pta
from collections import Counter
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


    def sell_multiple_indicators(self, last_candle, prev_candle) -> tuple:
        signals = []
        if last_candle['mom_14'] > last_candle['mom_14_upp']:
            signals.append('mom')
        if last_candle['rsi_14'] > 65:
            signals.append('rsi')
        if last_candle['mfi_14'] > 75:
            signals.append('mfi')
        if last_candle['cti_14'] > 0.75 and last_candle['cti_14'] <= prev_candle['cti_14']:
            signals.append( 'cti_20')
        if last_candle['cmf_14'] < 0.05 and last_candle['cmf_14'] <= prev_candle['cmf_14']:
            signals.append('cmf')
        if len(signals) >= 2:
            s = '_'.join(signals)
            return True, f"sell_{s}"

        return False, None

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = dataframe[dataframe['date'] >= trade_open_date]
        if df_trade.empty:
            return None

        candle_1 = dataframe.iloc[-1]
        candle_2 = dataframe.iloc[-2]

        max_close_candle = df_trade.nlargest(1, columns=['close'])
        min_close_candle = df_trade.nsmallest(1, columns=['close'])
        max_close = max_close_candle['close'].iloc[0]
        min_close = min_close_candle['close'].iloc[0]

        # use close instead of trade prices
        current_rate = candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        trade.max_rate = max_close
        trade.min_rate = min_close
        max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
        max_loss = (trade.open_rate - trade.min_rate) / trade.min_rate

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        # buy_tags = buy_tag.split()

        # do nothing with small losses or profits
        if not (-0.02 <= current_profit <= 0.02):
            sell, signal_name = self.sell_multiple_indicators(candle_1, candle_2)
            if sell:
                return f"{signal_name} ( {buy_tag})"

        return None

    def informative_1d_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1d: DataFrame = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1d)

        if self.config['runmode'].value not in ('live', 'dry_run'):
            if self.has_bt_agefilter:
                informative_1d['bt_agefilter'] = informative_1d['volume'].rolling(window=self.bt_min_age_days, min_periods=self.bt_min_age_days).count()

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1d_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1d

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h: DataFrame = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1h)

        # RSI
        informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)
        
        # Chopiness
        informative_1h['chop_84'] = qtpylib.chopiness(dataframe, 84)

        # Chaikin Money Flow
        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        # Williams %R
        informative_1h['r_480'] = williams_r(informative_1h, period=480)

        # CTI
        informative_1h['cti_20'] = pta.cti(informative_1h['close'], length=20)

        # CRSI (3, 2, 100)
        crsi_closechange = informative_1h['close'] / informative_1h['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative_1h['crsi'] = (ta.RSI(informative_1h['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(informative_1h['close'], 100)) / 3
        # CRSI (6, 3, 480)
        informative_1h['crsi_480'] = (ta.RSI(informative_1h['close'], timeperiod=6) + ta.RSI(crsi_updown, timeperiod=3) + ta.ROC(informative_1h['close'], 480)) / 3

        # CCI Oscillator
        cci_84: Series = ta.CCI(dataframe, timeperiod=84)
        cci_84_max: Series = cci_84.rolling(self.startup_candle_count).max()
        cci_84_min: Series = cci_84.rolling(self.startup_candle_count).min()
        informative_1h['cci_osc_84'] = (cci_84 / cci_84_max).where(cci_84 > 0, -cci_84 / cci_84_min)

        # MOMDIV
        mom = momdiv(informative_1h)
        informative_1h['momdiv_buy'] = mom['momdiv_buy']

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()

        # MFI
        dataframe['mfi_14'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['mfi_42'] = ta.MFI(dataframe, timeperiod=42)

        # CMF
        dataframe['cmf_14'] = chaikin_money_flow(dataframe, 14)
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        dataframe['cmf_42'] = chaikin_money_flow(dataframe, 42)

        # RSI
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_42'] = ta.RSI(dataframe, timeperiod=42)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_42'] = williams_r(dataframe, period=42)
        dataframe['r_480'] = williams_r(dataframe, period=480)

        # Stochastic RSI
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=96, fastk_period=3, fastd_period=3, fastd_matype=0)
        dataframe['stochrsi_fastk_96'] = stochrsi['fastk']
        dataframe['stochrsi_fastd_96'] = stochrsi['fastd']

        # CRSI
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi_100'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3
        dataframe['crsi_480'] = (ta.RSI(dataframe['close'], timeperiod=6) + ta.RSI(crsi_updown, timeperiod=3) + ta.ROC(dataframe['close'], 480)) / 3

        # CTI
        dataframe['cti_14'] = pta.cti(dataframe['close'], length=14)
        dataframe['cti_20'] = pta.cti(dataframe['close'], length=20)
        dataframe['cti_42'] = pta.cti(dataframe['close'], length=42)

        # CCI Oscillator
        cci_42: Series = ta.CCI(dataframe, timeperiod=42)
        cci_42_max: Series = cci_42.rolling(self.startup_candle_count).max()
        cci_42_min: Series = cci_42.rolling(self.startup_candle_count).min()
        dataframe['cci_osc_42'] = (cci_42 / cci_42_max).where(cci_42 > 0, -cci_42 / cci_42_min)

        # Momentum
        mom_14 = ta.MOM(dataframe, timeperiod=14)
        dataframe['mom_14'] = mom_14
        upperband, middleband, lowerband = ta.BBANDS(mom_14, timeperiod=42, nbdevup=2.0, nbdevdn=2.0, matype=0) # 0 = sma, 1 = ema, 2 = wma
        dataframe['mom_14_upp'] = upperband
        dataframe['mom_14_mid'] = middleband
        dataframe['mom_14_low'] = lowerband

        # Volume
        dataframe['volume_12'] = dataframe['volume'].rolling(12).sum()

        # Maximum positive and negative change in one hour
        dataframe['pump'] = dataframe[['open', 'close']].max(axis=1).rolling(window=12, min_periods=0).max() / dataframe[['open', 'close']].min(axis=1).shift(1).rolling(window=12, min_periods=0).min()
        dataframe['dump'] = dataframe[['open', 'close']].min(axis=1).rolling(window=12, min_periods=0).min() / dataframe[['open', 'close']].max(axis=1).shift(1).rolling(window=12, min_periods=0).max()

        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] normal_tf_indicators took: {tok - tik:0.4f} seconds.")

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        if self.info_timeframe_1d != 'none':
            informative_1d = self.informative_1d_indicators(dataframe, metadata)
            dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, self.info_timeframe_1d, ffill=True)
            drop_columns = [f"{s}_{self.info_timeframe_1d}" for s in ['date','open', 'high', 'low', 'close', 'volume']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        if self.info_timeframe_1h != 'none':
            informative_1h = self.informative_1h_indicators(dataframe, metadata)
            dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.info_timeframe_1h, ffill=True)
            drop_columns = [f"{s}_{self.info_timeframe_1h}" for s in ['date']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] Populate indicators took a total of: {tok - tik:0.4f} seconds.")

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''


        if conditions:
            dataframe.loc[:, 'buy'] = reduce(lambda x, y: x | y, conditions)

            df_buy: DataFrame = dataframe.loc[dataframe.loc[:, 'buy'], ['date', 'buy_tag', 'buy']]
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
                dataframe['pair'] = metadata['pair']
                with open(self.df_csv, 'a') as f:
                    dataframe.to_csv(f, sep='\t', header=f.tell()==0, index=False)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = False

        return dataframe

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
        candle_1: Series = df.iloc[-1]
        candle_2: Series = df.iloc[-2]
        buy_candle_date = candle_1['date']

        if buy_candle_date in self.custom_buy_info.keys():
            buy_tags = self.custom_buy_info[buy_candle_date][pair]
            # do not buy when there are many buy signals and concurrent buy tags
            if  self.max_concurrent_buy_signals > 0 and self.max_same_buy_tags > 0:
                buy_info = self.custom_buy_info[buy_candle_date]
                buy_signal_count = buy_info['buy_signals']
                buy_tag, buy_tag_count = Counter(buy_info['buy_tags'].split()).most_common()[0]
                if buy_signal_count > self.max_concurrent_buy_signals and buy_tag_count > self.max_same_buy_tags:
                    log.info(f"Buy for pair {pair} with buy tag {buy_tags}on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {self.max_concurrent_buy_signals}) and buy tag {buy_tag} was triggered {buy_tag_count} times (max = {self.max_same_buy_tags}).")
                    return False

            if self.config['runmode'].value not in ('live', 'dry_run'):
                close_price = candle_1['close']
                indicator = []
                indicator.append(candle_1['sma_75'])
                indicator.append(candle_1['ema_25'])
                indicator.append(candle_1['ewo_25'])
                indicator.append(candle_1['ewo'])
                indicator.append(candle_1['rsi_4'])
                indicator.append(candle_1['rsi_20'])
                indicator.append(candle_1['r_14'])
                indicator.append(candle_1['cmf_14'])
                indicator.append(candle_1['cti_14'])
                indicator.append(candle_1['rsi_14'])
                indicator.append(candle_1['mfi_14'])
                indicator.append(candle_1['cmf_42'])
                indicator.append(candle_1['cti_42'])
                indicator.append(candle_1['rsi_42'])
                indicator.append(candle_1['mfi_42'])
                indicator.append(candle_1['rsi_14_1h'])
                indicator.append(candle_1['crsi_100'])
                indicator.append(candle_1['crsi_480'])
                indicator.append(candle_1['crsi_1h'])
                indicator.append(candle_1['crsi_480_1h'])
                indicator.append(candle_1['cci_osc_42'])
                indicator.append(candle_1['cci_osc_84_1h'])
                indicator.append(candle_1['pump'])
                indicator.append(candle_1['dump'])
                indicator.append(candle_1['chop_84_1h'])
                indicator.append(candle_1['ema_vwma_osc_32'])
                indicator.append(candle_1['ema_vwma_osc_64'])
                indicator.append(candle_1['ema_vwma_osc_96'])
                indicator.append(candle_2['cmf_14'])
                indicator.append(candle_2['cti_14'])
                indicator.append(candle_2['rsi_14'])
                indicator.append(candle_2['mfi_14'])
                indicator.append(candle_2['cmf_42'])
                indicator.append(candle_2['cti_42'])
                indicator.append(candle_2['rsi_42'])
                indicator.append(candle_2['mfi_42'])
                indicator.append(candle_2['rsi_14_1h'])
                indicator.append(candle_2['crsi_100'])
                indicator.append(candle_2['crsi_480'])
                indicator.append(candle_2['crsi_1h'])
                indicator.append(candle_2['crsi_480_1h'])
                indicator.append(candle_2['cci_osc_42'])
                indicator.append(candle_2['cci_osc_84_1h'])
                with open(self.f_buys, 'a') as f:
                    print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};{close_price:.10n}", *indicator, sep=';', file=f)
        else:
            log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")

        return True

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # do not sell existing positions when there are many buy signals or concurrent buy tags
        if  self.max_concurrent_buy_signals > 0 or self.max_same_buy_tags > 0:
            candle_1: Series = df.iloc[-1]
            candle_date = candle_1['date']
            if candle_date in self.custom_buy_info.keys():
                buy_info = self.custom_buy_info[candle_date]
                if self.max_concurrent_buy_signals > 0:
                    buy_signal_count = buy_info['buy_signals']
                    if buy_signal_count > self.max_concurrent_buy_signals:
                        log.info(f"Sell for pair {pair} on candle {candle_date} with sell reason {sell_reason} is cancelled. There are {buy_signal_count} concurrent buy signals which is higher than the maximum ({self.max_concurrent_buy_signals}).")
                if self.max_same_buy_tags > 0:
                    buy_tag, buy_tag_count = Counter(buy_info['buy_tags'].split()).most_common()[0]
                    if buy_tag_count > self.max_same_buy_tags:
                        log.info(f"Sell for pair {pair} on candle {candle_date} with sell reason {sell_reason} is cancelled. Buy tag {buy_tag} was triggered {buy_tag_count} times which is higher than the maximum ({self.max_same_buy_tags}).")
                if buy_signal_count > self.max_concurrent_buy_signals or buy_tag_count > self.max_same_buy_tags:
                    return False

        if self.config['runmode'].value not in ('live', 'dry_run'):
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_close_date = timeframe_to_prev_date(self.timeframe, trade.close_date_utc)
            buy_tag = trade.buy_tag if trade is not None else 'empty'

            df_trade = df[(df['date'] >= trade_open_date) & (df['date'] <= trade_close_date)]
            if df_trade.empty:
                log.warning(f"confirm_trade_exit: Empty trade dataframe for pair {pair} on trade open date {trade_open_date}.")
                return False

            max_close_candle = df_trade.nlargest(1, columns=['close'])
            min_close_candle = df_trade.nsmallest(1, columns=['close'])
            min_close_date = min_close_candle['date'].to_numpy()[0]
            max_close_date = max_close_candle['date'].to_numpy()[0]
            profit = (rate - trade.open_rate) / trade.open_rate
            max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
            max_loss = (trade.min_rate - trade.open_rate) / trade.open_rate

            with open(self.f_trades, 'a') as f:
                print(f'{pair};{trade_open_date};{trade.open_rate:.10n};{trade_close_date};{rate:.10n};{buy_tag};{sell_reason.partition(" (")[0]};{profit:.10n};{max_profit:.10n};{max_loss:.10n};{trade.max_rate:.10n};{trade.min_rate:.10n};{max_close_date};{min_close_date};', file=f)

        return True


# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.SMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.SMA(dataframe, timeperiod=sma2_length)

    return ((sma1 - sma2) / dataframe['close']) * 100


# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)

    return cmf


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe['high'].rolling(period).max()
    lowest_low = dataframe['low'].rolling(period).min()

    WR = Series((highest_high - dataframe['close']) / (highest_high - lowest_low))

    return WR * -100
    

# Volume Weighted Simple Moving Average
def vwma(dataframe: DataFrame, length: int = 10) -> Series:
    pv = dataframe['close'] * dataframe['volume']

    return Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))

# this is not a Modified Elder Ray Index
def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))

    return slow_ma > slow_ma.shift(1)  # we just need true & false for ERI trend

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))

    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def ema_sma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(ta.SMA(dataframe, len_slow_ma), len_slow_ma))

    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def zlema(dataframe, timeperiod):
    lag = round((timeperiod - 1) / 2)
    if isinstance(dataframe, Series):
        ema_data = dataframe + (dataframe - dataframe.shift(lag))
    else:
        ema_data = dataframe['close'] + (dataframe['close'] - dataframe['close'].shift(lag))

    return ta.EMA(ema_data, timeperiod)


def hull(dataframe, timeperiod):
    if isinstance(dataframe, Series):
        return ta.WMA(2 * ta.WMA(dataframe, round(timeperiod / 2)) - ta.WMA(dataframe, timeperiod), round(np.sqrt(timeperiod)))
    else:
        return ta.WMA(2 * ta.WMA(dataframe['close'], round(timeperiod / 2)) - ta.WMA(dataframe['close'], timeperiod), round(np.sqrt(timeperiod)))


# PMAX
# def pmax(df, period, multiplier, length, MAtype, src):

#     period = int(period)
#     multiplier = int(multiplier)
#     length = int(length)
#     MAtype = int(MAtype)
#     src = int(src)

#     mavalue = 'MA_' + str(MAtype) + '_' + str(length)
#     atr = 'ATR_' + str(period)
#     pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
#     pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)

#     # MAtype==1 --> EMA
#     # MAtype==2 --> DEMA
#     # MAtype==3 --> T3
#     # MAtype==4 --> SMA
#     # MAtype==5 --> VIDYA
#     # MAtype==6 --> TEMA
#     # MAtype==7 --> WMA
#     # MAtype==8 --> VWMA
#     # MAtype==9 --> zema
    
#     if src == 1:
#         masrc = df['close']
#     elif src == 2:
#         masrc = (df['high'] + df['low']) / 2
#     elif src == 3:
#         masrc = (df['high'] + df['low'] + df['close'] + df['open']) / 4

#     if MAtype == 1:
#         mavalue = ta.EMA(masrc, timeperiod=length)
#     elif MAtype == 2:
#         mavalue = ta.DEMA(masrc, timeperiod=length)
#     elif MAtype == 3:
#         mavalue = ta.T3(masrc, timeperiod=length)
#     elif MAtype == 4:
#         mavalue = ta.SMA(masrc, timeperiod=length)
#     elif MAtype == 5:
#         mavalue = VIDYA(df, length=length)
#     elif MAtype == 6:
#         mavalue = ta.TEMA(masrc, timeperiod=length)
#     elif MAtype == 7:
#         mavalue = ta.WMA(df, timeperiod=length)
#     elif MAtype == 8:
#         mavalue = vwma(df, length)
#     elif MAtype == 9:
#         mavalue = zlema(df, length)

#     df[atr] = ta.ATR(df, timeperiod=period)
#     df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
#     df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])

#     basic_ub = df['basic_ub'].values
#     final_ub = np.full(len(df), 0.00)
#     basic_lb = df['basic_lb'].values
#     final_lb = np.full(len(df), 0.00)

#     for i in range(period, len(df)):
#         final_ub[i] = basic_ub[i] if (
#             basic_ub[i] < final_ub[i - 1]
#             or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
#         final_lb[i] = basic_lb[i] if (
#             basic_lb[i] > final_lb[i - 1]
#             or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

#     df['final_ub'] = final_ub
#     df['final_lb'] = final_lb

#     pm_arr = np.full(len(df), 0.00)
#     for i in range(period, len(df)):
#         pm_arr[i] = (
#             final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
#                                     and mavalue[i] <= final_ub[i])
#         else final_lb[i] if (
#             pm_arr[i - 1] == final_ub[i - 1]
#             and mavalue[i] > final_ub[i]) else final_lb[i]
#         if (pm_arr[i - 1] == final_lb[i - 1]
#             and mavalue[i] >= final_lb[i]) else final_ub[i]
#         if (pm_arr[i - 1] == final_lb[i - 1]
#             and mavalue[i] < final_lb[i]) else 0.00)

#     pm = Series(pm_arr)

#     # Mark the trend direction up/down
#     pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

#     return pm, pmx


def SSLChannels(dataframe: DataFrame, length = 7):
    ATR = ta.ATR(dataframe, timeperiod=14)
    smaHigh = dataframe['high'].rolling(length).mean() + ATR
    smaLow = dataframe['low'].rolling(length).mean() - ATR
    hlv = Series(np.where(dataframe['close'] > smaHigh, 1, np.where(dataframe['close'] < smaLow, -1, np.NAN)))
    hlv = hlv.ffill()
    sslDown = np.where(hlv < 0, smaHigh, smaLow)
    sslUp = np.where(hlv < 0, smaLow, smaHigh)

    return sslDown, sslUp


def pivot_points(dataframe: DataFrame, mode = 'fibonacci') -> Series:
    hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
    hl_range = (dataframe['high'] - dataframe['low']).shift(1)
    if mode == 'simple':
        res1 = hlc3_pivot * 2 - dataframe['low'].shift(1)
        sup1 = hlc3_pivot * 2 - dataframe['high'].shift(1)
        res2 = hlc3_pivot + (dataframe['high'] - dataframe['low']).shift()
        sup2 = hlc3_pivot - (dataframe['high'] - dataframe['low']).shift()
        res3 = hlc3_pivot * 2 + (dataframe['high'] - 2 * dataframe['low']).shift()
        sup3 = hlc3_pivot * 2 - (2 * dataframe['high'] - dataframe['low']).shift()
    elif mode == 'fibonacci':
        res1 = hlc3_pivot + 0.382 * hl_range
        sup1 = hlc3_pivot - 0.382 * hl_range
        res2 = hlc3_pivot + 0.618 * hl_range
        sup2 = hlc3_pivot - 0.618 * hl_range
        res3 = hlc3_pivot + 1 * hl_range
        sup3 = hlc3_pivot - 1 * hl_range

    return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3


def HeikinAshi(dataframe, smooth_inputs = False, smooth_outputs = False, length = 10):
    df = dataframe[['open','close','high','low']].copy().fillna(0)
    if smooth_inputs:
        df['open_s']  = ta.EMA(df['open'], timeframe = length)
        df['high_s']  = ta.EMA(df['high'], timeframe = length)
        df['low_s']   = ta.EMA(df['low'],  timeframe = length)
        df['close_s'] = ta.EMA(df['close'],timeframe = length)
        open_ha  = (df['open_s'].shift(1) + df['close_s'].shift(1)) / 2
        high_ha  = df.loc[:, ['high_s', 'open_s', 'close_s']].max(axis=1)
        low_ha   = df.loc[:, ['low_s', 'open_s', 'close_s']].min(axis=1)
        close_ha = (df['open_s'] + df['high_s'] + df['low_s'] + df['close_s'])/4
    else:
        open_ha  = (df['open'].shift(1) + df['close'].shift(1)) / 2
        high_ha  = df.loc[:, ['high', 'open', 'close']].max(axis=1)
        low_ha   = df.loc[:, ['low', 'open', 'close']].min(axis=1)
        close_ha = (df['open'] + df['high'] + df['low'] + df['close'])/4
    open_ha = open_ha.fillna(0)
    high_ha = high_ha.fillna(0)
    low_ha  = low_ha.fillna(0)
    close_ha = close_ha.fillna(0)
    if smooth_outputs:
        open_sha  = ta.EMA(open_ha, timeframe = length)
        high_sha  = ta.EMA(high_ha, timeframe = length)
        low_sha   = ta.EMA(low_ha, timeframe = length)
        close_sha = ta.EMA(close_ha, timeframe = length)
        return open_sha, close_sha, low_sha
    else:
        return open_ha, close_ha, low_ha


# Mom DIV
def momdiv(dataframe: DataFrame, mom_length: int = 10, bb_length: int = 20, bb_dev: float = 2.0, lookback: int = 30) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    # hh = dataframe['high'].rolling(lookback).max()
    # ll = dataframe['low'].rolling(lookback).min()
    # coh = dataframe['high'] >= hh
    # col = dataframe['low'] <= ll
    df = DataFrame({
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            # "momdiv_sell": sell,
            # "momdiv_coh": coh,
            # "momdiv_col": col,
        }, index=dataframe['close'].index)
    return df


class MyNFI(NFI):

    f_buys = './user_data/mynfibuys.txt'
    f_trades = './user_data/mynfitrades.txt'

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_1 = df.iloc[-1]
        candle_2 = df.iloc[-2]

        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        buy_tag = trade.buy_tag if trade is not None else 'empty'
        buy_tags = buy_tag.split()

        df_trade: DataFrame = df.loc[df['date'] >= trade_open_date]
        if df_trade.empty:
            log.warning(f"Empty trade dataframe for pair {pair} on candle {df['date']}.")
            return None

        max_close_candle = df_trade.nlargest(1, columns=['close'])
        min_close_candle = df_trade.nsmallest(1, columns=['close'])
        max_close = max_close_candle['close'].iloc[0]
        min_close = min_close_candle['close'].iloc[0]

        # use close instead of trade prices
        current_rate = candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        trade.max_rate = max_close
        trade.min_rate = min_close

        max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
        max_loss = (trade.open_rate - trade.min_rate) / trade.min_rate

        # i = number of candles between trade and last candle
        # calculate good indicators on the fly for number of candles since lowest point and/or highest point

        # broken protections
        if (
            current_profit < -0.04
            and len(df) >= 72
            and len(df_trade) >= 36
        ):
            candle_12 = df.iloc[-12]
            candle_20 = df.iloc[-20]
            candle_24 = df.iloc[-24]
            candle_36 = df.iloc[-36]

            for b in buy_tags:
                i = int(b)
                item_sell_logic = []
                protection_params = self.buy_protection_params[i]
                if self.buy_params[f"buy_condition_{i}_enable"]:
                    if protection_params["ema_fast"]:
                        item_sell_logic.append(candle_1[f"ema_{protection_params['ema_fast_len']}"] < candle_1['ema_200'])
                    if protection_params["ema_slow"]:
                        item_sell_logic.append(candle_1[f"ema_{protection_params['ema_slow_len']}_1h"] < candle_1['ema_200_1h'])
                    if protection_params["close_above_ema_fast"]:
                        item_sell_logic.append(candle_1['close'] < candle_1[f"ema_{protection_params['close_above_ema_fast_len']}"])
                    if protection_params["close_above_ema_slow"]:
                        item_sell_logic.append(candle_1['close'] < candle_1[f"ema_{protection_params['close_above_ema_slow_len']}_1h"])
                    if protection_params["sma200_rising"]:
                        item_sell_logic.append(candle_1['sma_200'] < df['sma_200'].iloc[-int(protection_params['sma200_rising_val'])])
                    if protection_params["sma200_1h_rising"]:
                        item_sell_logic.append(candle_1['sma_200_1h'] < df['sma_200_1h'].iloc[-int(protection_params['sma200_1h_rising_val'])])

                    if i in (5, 6, 7):
                        item_sell_logic.append(candle_1['ema_26'] < candle_1['ema_12'])
                    if i == 8:
                        item_sell_logic.append(candle_1['ema_20'] < candle_1['ema_50'])
                        item_sell_logic.append(candle_1['ema_15'] < candle_1['ema_100'])
                        item_sell_logic.append(candle_1['ema_200'] < candle_1['sma_200'])
                    if i == 9:
                        item_sell_logic.append(candle_1['ema_50'] < candle_1['ema_200'])
                    if i in (10, 13):
                        item_sell_logic.append(candle_1['ema_50_1h'] < candle_1['ema_100_1h'])
                    if i == 18:
                        item_sell_logic.append(candle_1['sma_200'] < candle_20['sma_200'])
                        item_sell_logic.append(candle_1['sma_200_1h'] < candle_36['sma_200_1h'])
                    if i == 19:
                        item_sell_logic.append(candle_1['ema_vwma_osc_32'] < 0)
                        item_sell_logic.append(candle_1['ema_vwma_osc_64'] < 0)
                        item_sell_logic.append(candle_1['ema_vwma_osc_96'] < 0)
                    if i == 22:
                        item_sell_logic.append(candle_1['ema_100'] < candle_12['ema_100_1h'])
                        item_sell_logic.append(candle_1['ema_200_1h'] < candle_36['ema_200_1h'])
                    if i == 24:
                        item_sell_logic.append(candle_1['ema_12_1h'] < candle_1['ema_35_1h'])
                    if i == 25:
                        item_sell_logic.append(candle_1['ema_20_1h'] < candle_1['ema_26_1h'])
                    if i in (28, 29):
                        item_sell_logic.append(candle_1['ema_vwma_osc_64'] < 0)
                    if i == 37:
                        item_sell_logic.append(candle_1['ema_vwma_osc_96'] < 0)
                    if i in (41, 42, 43):
                        item_sell_logic.append(candle_1['ema_200_1h'] < candle_12['ema_200_1h'])
                        item_sell_logic.append(candle_12['ema_200_1h'] < candle_24['ema_200_1h'])
                    if i == 48:
                        item_sell_logic.append(candle_1['ema_200_1h'] < candle_12['ema_200'])
                        item_sell_logic.append(candle_12['ema_200_1h'] < candle_24['ema_200_1h'])
                        item_sell_logic.append(candle_1['ema_vwma_osc_32'] < 0)
                        item_sell_logic.append(candle_1['ema_vwma_osc_64'] < 0)
                        item_sell_logic.append(candle_1['ema_vwma_osc_96'] < 0)

                    if item_sell_logic:
                        item_sell = reduce(lambda x, y: x & y, item_sell_logic)
                        if item_sell:
                            return f"protection_broken ( {buy_tag})"

        # stoploss
        # sell, signal_name = self.sell_stoploss(current_profit, max_profit, max_loss, candle_1, candle_2, trade, current_time)
        # if sell:
        #     return f"{signal_name} ( {buy_tag})"

        # do nothing with small losses or profits
        if not (-0.02 <= current_profit <= 0.02):
            sell, signal_name = self.sell_multiple_indicators(candle_1, candle_2)
            if sell:
                return f"{signal_name} ( {buy_tag})"

        return None


class NFI1(NFI):

    b = __name__.lower()
    if b not in ('nfi', 'mynfi'):
        f_buys = f"./user_data/{b}buys.txt"
        f_trades = f"./user_data/{b}trades.txt"
        n = int("".join(filter(str.isdigit, b)))

        # Maximum number of concurrent buy signals (0 is disable)
        max_concurrent_buy_signals = 0
        # Maximum number of buys with the same buy tag (0 is disable)
        max_same_buy_tags = 0

        buy_params = {}
        for i in range(1, 49):
            buy_params[f"buy_condition_{i}_enable"] = False

        buy_params[f"buy_condition_{n}_enable"] = True


class NFI2(NFI1):
    pass
class NFI3(NFI1):
    pass
class NFI4(NFI1):
    pass
class NFI5(NFI1):
    pass
class NFI6(NFI1):
    pass
class NFI7(NFI1):
    pass
class NFI8(NFI1):
    pass
class NFI9(NFI1):
    pass
class NFI10(NFI1):
    pass
class NFI11(NFI1):
    pass
class NFI12(NFI1):
    pass
class NFI13(NFI1):
    pass
class NFI14(NFI1):
    pass
class NFI15(NFI1):
    pass
class NFI16(NFI1):
    pass
class NFI17(NFI1):
    pass
class NFI18(NFI1):
    pass
class NFI19(NFI1):
    pass
class NFI20(NFI1):
    pass
class NFI21(NFI1):
    pass
class NFI22(NFI1):
    pass
class NFI23(NFI1):
    pass
class NFI24(NFI1):
    pass
class NFI25(NFI1):
    pass
class NFI26(NFI1):
    pass
class NFI27(NFI1):
    pass
class NFI28(NFI1):
    pass
class NFI29(NFI1):
    pass
class NFI30(NFI1):
    pass
class NFI31(NFI1):
    pass
class NFI32(NFI1):
    pass
class NFI33(NFI1):
    pass
class NFI34(NFI1):
    pass
class NFI35(NFI1):
    pass
class NFI36(NFI1):
    pass
class NFI37(NFI1):
    pass
class NFI38(NFI1):
    pass
class NFI39(NFI1):
    pass
class NFI40(NFI1):
    pass
class NFI41(NFI1):
    pass
class NFI42(NFI1):
    pass
class NFI43(NFI1):
    pass
class NFI44(NFI1):
    pass
class NFI45(NFI1):
    pass
class NFI46(NFI1):
    pass
class NFI47(NFI1):
    pass
class NFI48(NFI1):
    pass
class NFI49(NFI1):
    pass


class NFIAll(NFI):

    b = __name__.lower()
    f_buys = f"./user_data/{b}buys.txt"
    f_trades = f"./user_data/{b}trades.txt"

    # Maximum number of concurrent buy signals (0 is disable)
    max_concurrent_buy_signals = 0
    # Maximum number of buys with the same buy tag (0 is disable)
    max_same_buy_tags = 0

    buy_params = {}
    for i in range(1, 49):
        buy_params[f"buy_condition_{i}_enable"] = True
