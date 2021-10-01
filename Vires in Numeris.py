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

    stoploss = -0.50

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    res_timeframe = 'none'
    info_timeframe_1h = '1h'
    info_timeframe_1d = '1d'

    # BTC informative
    has_BTC_base_tf = False
    has_BTC_info_tf = True
    has_BTC_daily_tf = False

    # Backtest Age Filter emulation
    has_bt_agefilter = True
    bt_min_age_days = 21

    # Exchange Downtime protection
    has_downtime_protection = False

    # Do you want to use the hold feature? (with hold-trades.json)
    holdSupportEnabled = False

    # Coin Metrics
    coin_metrics = {}
    coin_metrics['top_traded_enabled'] = False
    coin_metrics['top_traded_updated'] = False
    coin_metrics['top_traded_len'] = 10
    coin_metrics['tt_dataframe'] = DataFrame()
    coin_metrics['top_grossing_enabled'] = False
    coin_metrics['top_grossing_updated'] = False
    coin_metrics['top_grossing_len'] = 20
    coin_metrics['tg_dataframe'] = DataFrame()
    coin_metrics['current_whitelist'] = []

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
    startup_candle_count: int = 480

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'trailing_stop_loss': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    #############################################################

    buy_params = {
        #############
        # Enable/Disable conditions
        "buy_condition_1_enable": False,
        "buy_condition_2_enable": True,
        "buy_condition_3_enable": False,
        "buy_condition_4_enable": True,
        "buy_condition_5_enable": True,
        "buy_condition_6_enable": True,
        "buy_condition_7_enable": False,
        "buy_condition_8_enable": False,
        "buy_condition_9_enable": False,
        "buy_condition_10_enable": True,
        "buy_condition_11_enable": False,
        "buy_condition_12_enable": True,
        "buy_condition_13_enable": True,
        "buy_condition_14_enable": True,
        "buy_condition_15_enable": True,
        "buy_condition_16_enable": True,
        "buy_condition_17_enable": False,
        "buy_condition_18_enable": False,
        "buy_condition_19_enable": False,
        "buy_condition_20_enable": True,
        "buy_condition_21_enable": False,
        "buy_condition_22_enable": True,
        "buy_condition_23_enable": True,
        "buy_condition_24_enable": False,
        "buy_condition_25_enable": False,
        "buy_condition_26_enable": True,
        "buy_condition_27_enable": False,
        "buy_condition_28_enable": True,
        "buy_condition_29_enable": False,
        "buy_condition_30_enable": True,
        "buy_condition_31_enable": False,
        "buy_condition_32_enable": False,
        "buy_condition_33_enable": True,
        "buy_condition_34_enable": True,
        "buy_condition_35_enable": False,
        "buy_condition_36_enable": False,
        "buy_condition_37_enable": True,
        "buy_condition_38_enable": True,
        "buy_condition_39_enable": False,
        "buy_condition_40_enable": False,
        "buy_condition_41_enable": False,
        "buy_condition_42_enable": True,
        "buy_condition_43_enable": True,
        "buy_condition_44_enable": True,
        "buy_condition_45_enable": True,
        "buy_condition_46_enable": False,
        "buy_condition_47_enable": True,
        "buy_condition_48_enable": False,
    #############
    }

    sell_params = {
        #############
        # Enable/Disable conditions
        "sell_condition_1_enable": True,
        "sell_condition_2_enable": True,
        "sell_condition_3_enable": True,
        "sell_condition_4_enable": True,
        "sell_condition_5_enable": True,
        "sell_condition_6_enable": True,
        "sell_condition_7_enable": True,
        "sell_condition_8_enable": True,
        #############
    }

    #############################################################

    buy_protection_params = {
        1: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "26",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "28",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "70",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        2: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "20",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "48",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "20",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.4
        },
        3: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "100",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "36",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : True,
            "safe_pump_type"            : "110",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        4: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "110",
            "safe_pump_period"          : "48",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        5: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "100",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "100",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        6: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        7: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "100",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "80",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        8: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "12",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : True,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "36",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.016,
            "safe_dips_threshold_2"     : 0.11,
            "safe_dips_threshold_12"    : 0.26,
            "safe_dips_threshold_144"   : 0.44,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.05
        },
        9: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "100",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.1
        },
        10: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "35",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "24",
            "safe_dips_threshold_0"     : 0.016,
            "safe_dips_threshold_2"     : 0.11,
            "safe_dips_threshold_12"    : 0.26,
            "safe_dips_threshold_144"   : 0.44,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.6
        },
        11: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "20",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "24",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "36",
            "safe_dips_threshold_0"     : 0.022,
            "safe_dips_threshold_2"     : 0.18,
            "safe_dips_threshold_12"    : 0.34,
            "safe_dips_threshold_144"   : 0.56,
            "safe_pump"                 : False,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        12: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "24",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.3
        },
        13: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "24",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "50",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        14: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : True,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.5
        },
        15: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "80",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        16: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "50",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.027,
            "safe_dips_threshold_2"     : 0.26,
            "safe_dips_threshold_12"    : 0.44,
            "safe_dips_threshold_144"   : 0.84,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        17: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        18: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "100",
            "ema_slow"                  : True,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : True,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : True,
            "sma200_rising_val"         : "44",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "72",
            "safe_dips_threshold_0"     : 0.026,
            "safe_dips_threshold_2"     : 0.24,
            "safe_dips_threshold_12"    : 0.42,
            "safe_dips_threshold_144"   : 0.8,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        19: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "36",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "36",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "50",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        20: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "50",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        21: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.025,
            "safe_dips_threshold_2"     : 0.23,
            "safe_dips_threshold_12"    : 0.4,
            "safe_dips_threshold_144"   : 0.7,
            "safe_pump"                 : False,
            "safe_pump_type"            : "50",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        22: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "50",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "110",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.6
        },
        23: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "15",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : True,
            "sma200_rising_val"         : "24",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.022,
            "safe_dips_threshold_2"     : 0.1,
            "safe_dips_threshold_12"    : 0.3,
            "safe_dips_threshold_144"   : 0.84,
            "safe_pump"                 : True,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        24: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "50",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "36",
            "safe_dips_threshold_0"     : 0.016,
            "safe_dips_threshold_2"     : 0.11,
            "safe_dips_threshold_12"    : 0.26,
            "safe_dips_threshold_144"   : 0.44,
            "safe_pump"                 : False,
            "safe_pump_type"            : "10",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        25: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "20",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "36",
            "safe_dips_threshold_0"     : 0.024,
            "safe_dips_threshold_2"     : 0.22,
            "safe_dips_threshold_12"    : 0.38,
            "safe_dips_threshold_144"   : 0.66,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "pivot", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 0.98,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.4
        },
        26: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "100",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.016,
            "safe_dips_threshold_2"     : 0.1,
            "safe_dips_threshold_12"    : 0.11,
            "safe_dips_threshold_144"   : 0.22,
            "safe_pump"                 : True,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.35
        },
        27: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "50",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        28: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 0.99,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.32
        },
        29: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "110",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "pivot", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.01
        },
        30: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "110",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        31: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.02,
            "safe_dips_threshold_2"     : 0.14,
            "safe_dips_threshold_12"    : 0.32,
            "safe_dips_threshold_144"   : 0.5,
            "safe_pump"                 : False,
            "safe_pump_type"            : "10",
            "safe_pump_period"          : "48",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "sup3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 0.98,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        32: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "80",
            "safe_pump_period"          : "48",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        33: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        34: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "10",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 0.99,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        35: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.1
        },
        36: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "10",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        37: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "48",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.5
        },
        38: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : False,
            "ema_slow_len"              : "100",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "50",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "100",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "50",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "10",
            "safe_pump_period"          : "36",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        39: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "100",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "100",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "50",
            "safe_pump_period"          : "48",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        40: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : True,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : True,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "48",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.2
        },
        41: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.015,
            "safe_dips_threshold_2"     : 0.1,
            "safe_dips_threshold_12"    : 0.24,
            "safe_dips_threshold_144"   : 0.42,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        42: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "12",
            "ema_slow"                  : False,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.027,
            "safe_dips_threshold_2"     : 0.26,
            "safe_dips_threshold_12"    : 0.44,
            "safe_dips_threshold_144"   : 0.84,
            "safe_pump"                 : True,
            "safe_pump_type"            : "10",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        43: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "12",
            "ema_slow"                  : False,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.024,
            "safe_dips_threshold_2"     : 0.22,
            "safe_dips_threshold_12"    : 0.38,
            "safe_dips_threshold_144"   : 0.66,
            "safe_pump"                 : False,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        44: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "12",
            "ema_slow"                  : False,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        45: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "15",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "20",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.3,
            "safe_dips_threshold_12"    : 0.48,
            "safe_dips_threshold_144"   : 0.9,
            "safe_pump"                 : False,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        46: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "50",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "20",
            "safe_dips_threshold_0"     : 0.028,
            "safe_dips_threshold_2"     : 0.06,
            "safe_dips_threshold_12"    : 0.25,
            "safe_dips_threshold_144"   : 0.26,
            "safe_pump"                 : False,
            "safe_pump_type"            : "100",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : True,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "res3", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 2.0
        },
        47: {
            "ema_fast"                  : False,
            "ema_fast_len"              : "12",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : False,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : False,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : False,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : False,
            "sma200_1h_rising_val"      : "24",
            "safe_dips_threshold_0"     : 0.025,
            "safe_dips_threshold_2"     : 0.05,
            "safe_dips_threshold_12"    : 0.25,
            "safe_dips_threshold_144"   : 0.5,
            "safe_pump"                 : True,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0
        },
        48: {
            "ema_fast"                  : True,
            "ema_fast_len"              : "12",
            "ema_slow"                  : True,
            "ema_slow_len"              : "12",
            "close_above_ema_fast"      : True,
            "close_above_ema_fast_len"  : "200",
            "close_above_ema_slow"      : True,
            "close_above_ema_slow_len"  : "200",
            "sma200_rising"             : True,
            "sma200_rising_val"         : "30",
            "sma200_1h_rising"          : True,
            "sma200_1h_rising_val"      : "24",
            "safe_dips_threshold_0"     : None,
            "safe_dips_threshold_2"     : None,
            "safe_dips_threshold_12"    : None,
            "safe_dips_threshold_144"   : None,
            "safe_pump"                 : False,
            "safe_pump_type"            : "120",
            "safe_pump_period"          : "24",
            "btc_1h_not_downtrend"      : False,
            "close_over_pivot_type"     : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_over_pivot_offset"   : 1.0,
            "close_under_pivot_type"    : "none", # pivot, sup1, sup2, sup3, res1, res2, res3
            "close_under_pivot_offset"  : 1.0

        }
    }    

    buy_1_min_inc = 0.022
    buy_1_rsi_max = 32.0
    buy_1_r_14_max = -75.0
    buy_1_mfi_max = 46.0
    buy_1_rsi_1h_min = 30.0
    buy_1_rsi_1h_max = 84.0

    buy_2_rsi_1h_diff = 39.0
    buy_2_mfi = 49.0
    buy_2_cti_20_max = -0.9
    buy_2_r_480_min = -95.0
    buy_2_r_480_max = -46.0
    buy_2_cti_20_1h_max = 0.9
    buy_2_volume = 2.0

    buy_3_bb40_bbdelta_close = 0.057
    buy_3_bb40_closedelta_close = 0.023
    buy_3_bb40_tail_bbdelta = 0.418
    buy_3_cti_20_max = -0.5
    buy_3_cci_osc_42_min = -0.25
    buy_3_crsi_1h_min = 20.0
    buy_3_r_480_1h_min = -48.0
    buy_3_cti_20_1h_max = 0.82

    buy_4_bb20_close_bblowerband = 0.98
    buy_4_bb20_volume = 10.0
    buy_4_cti_20_max = -0.8

    buy_5_ema_rel = 0.84
    buy_5_ema_open_mult = 0.02
    buy_5_bb_offset = 0.999
    buy_5_cti_20_max = -0.5
    buy_5_r_14_max = -94.0
    buy_5_rsi_14_min = 25.0
    buy_5_mfi_min = 18.0
    buy_5_crsi_1h_min = 12.0
    buy_5_volume = 1.6

    buy_6_ema_open_mult = 0.019
    buy_6_bb_offset = 0.984
    buy_6_r_14_max = -85.0
    buy_6_crsi_1h_min = 15.0
    buy_6_cti_20_1h_min = 0.0

    buy_7_ema_open_mult = 0.031
    buy_7_ma_offset = 0.978
    buy_7_cti_20_max = -0.9
    buy_7_rsi_max = 45.0

    buy_8_bb_offset = 0.986
    buy_8_r_14_max = -98.0
    buy_8_cti_20_1h_max = 0.95
    buy_8_r_480_1h_max = -18.0
    buy_8_volume = 1.8

    buy_9_ma_offset = 0.968
    buy_9_bb_offset = 0.982
    buy_9_mfi_max = 50.0
    buy_9_cti_20_max = -0.85
    buy_9_r_14_max = -94.0
    buy_9_rsi_1h_min = 20.0
    buy_9_rsi_1h_max = 88.0
    buy_9_crsi_1h_min = 21.0

    buy_10_ma_offset_high = 0.94
    buy_10_bb_offset = 0.984
    buy_10_r_14_max = -88.0
    buy_10_cti_20_1h_min = -0.5
    buy_10_cti_20_1h_max = 0.94

    buy_11_ma_offset = 0.956
    buy_11_min_inc = 0.022
    buy_11_rsi_max = 37.0
    buy_11_mfi_max = 46.0
    buy_11_cci_max = -120.0
    buy_11_cci_osc_42_max = -0.40
    buy_11_r_480_max = -32.0
    buy_11_rsi_1h_min = 30.0
    buy_11_rsi_1h_max = 84.0
    buy_11_cti_20_1h_max = 0.91
    buy_11_r_480_1h_max = -25.0
    buy_11_crsi_1h_min = 26.0

    buy_12_ma_offset = 0.927
    buy_12_ewo_min = 2.0
    buy_12_rsi_max = 32.0
    buy_12_cti_20_max = -0.9

    buy_13_ma_offset = 0.99
    buy_13_cti_20_max = -0.92
    buy_13_ewo_max = -6.0
    buy_13_cti_20_1h_max = -0.88
    buy_13_crsi_1h_min = 10.0

    buy_14_ema_open_mult = 0.014
    buy_14_bb_offset = 0.989
    buy_14_ma_offset = 0.945
    buy_14_cti_20_max = -0.85

    buy_15_ema_open_mult = 0.0238
    buy_15_ma_offset = 0.958
    buy_15_rsi_min = 28.0
    buy_15_cti_20_1h_min = -0.2

    buy_16_ma_offset = 0.942
    buy_16_ewo_min = 2.0
    buy_16_rsi_max = 36.0
    buy_16_cti_20_max = -0.9

    buy_17_ma_offset = 0.999
    buy_17_ewo_max = -7.0
    buy_17_cti_20_max = -0.96
    buy_17_crsi_1h_min = 12.0
    buy_17_volume = 2.0

    buy_18_bb_offset = 0.986
    buy_18_rsi_max = 33.5
    buy_18_cti_20_max = -0.85
    buy_18_cti_20_1h_max = 0.91
    buy_18_volume = 2.0

    buy_19_rsi_1h_min = 30.0
    buy_19_chop_max = 21.3

    buy_20_rsi_14_max = 36.0
    buy_20_rsi_14_1h_max = 16.0
    buy_20_cti_20_max = -0.84
    buy_20_volume = 2.0

    buy_21_rsi_14_max = 14.0
    buy_21_rsi_14_1h_max = 28.0
    buy_21_cti_20_max = -0.902
    buy_21_volume = 2.0

    buy_22_volume = 2.0
    buy_22_bb_offset = 0.984
    buy_22_ma_offset = 0.98
    buy_22_ewo_min = 5.6
    buy_22_rsi_14_max = 36.0
    buy_22_cti_20_max = -0.54
    buy_22_r_480_max = -40.0
    buy_22_cti_20_1h_min = -0.5

    buy_23_bb_offset = 0.984
    buy_23_ewo_min = 3.4
    buy_23_rsi_14_max = 28.0
    buy_23_cti_20_max = -0.74
    buy_23_rsi_14_1h_max = 80.0
    buy_23_r_480_1h_min = -95.0
    buy_23_cti_20_1h_max = 0.92

    buy_24_rsi_14_max = 50.0
    buy_24_rsi_14_1h_min = 66.9

    buy_25_ma_offset = 0.953
    buy_25_rsi_4_max = 30.0
    buy_25_cti_20_max = -0.78
    buy_25_cci_max = -200.0
    buy_25_cci_osc_42_max = -0.68

    buy_26_zema_low_offset = 0.9405
    buy_26_cti_20_max = -0.72
    buy_26_cci_max = -166.0
    buy_26_r_14_max = -98.0
    buy_26_cti_20_1h_max = 0.95
    buy_26_volume = 2.0

    buy_27_wr_max = -95.0
    buy_27_r_14 = -100.0
    buy_27_wr_1h_max = -90.0
    buy_27_rsi_max = 46.0
    buy_27_volume = 2.0

# from here is ewo
# few good trades
    buy_28_ma_offset = 0.928
    buy_28_ewo_min = 2.0
    buy_28_rsi_14_max = 33.4
    buy_28_cti_20_max = -0.84
    buy_28_r_14_max = -97.0
    buy_28_cti_20_1h_max = 0.95


    buy_29_ma_offset = 0.984
    buy_29_ewo_max = -4.2
    buy_29_cti_20_max = -0.96

    buy_30_ma_offset = 0.962
    buy_30_ewo_min = 6.4
    buy_30_ewo_max = 8.5
    buy_30_rsi_14_max = 34.0
    buy_30_cti_20_max = -0.87
    buy_30_r_14_max = -97.0

    buy_31_ma_offset = 0.962
    buy_31_ewo_max = -5.2
    buy_31_r_14_max = -94.0
    buy_31_cti_20_max = -0.9

# no ewo but loosing
    buy_32_ma_offset = 0.942
    buy_32_rsi_4_max = 46.0
    buy_32_cti_20_max = -0.86
    buy_32_rsi_14_min = 19.0
    buy_32_crsi_1h_min = 10.0
    buy_32_crsi_1h_max = 60.0

# combine with 37
    buy_33_ma_offset = 0.988
    buy_33_ewo_min = 9.0
    buy_33_rsi_max = 32.0
    buy_33_cti_20_max = -0.88
    buy_33_r_14_max = -98.0
    buy_33_cti_20_1h_max = 0.92
    buy_33_volume = 2.0

# combine with 38, create two with negative ewo?
    buy_34_ma_offset = 0.97
    buy_34_ewo_max = -4.0
    buy_34_cti_20_max = -0.95
    buy_34_r_14_max = -99.9
    buy_34_crsi_1h_min = 8.0
    buy_34_volume = 2.0

# old 35
    buy_35_ma_offset = 0.984
    buy_35_ewo_min = 7.8
    buy_35_rsi_max = 32.0
    buy_35_cti_20_max = -0.8
    buy_35_r_14_max = -95.0

# old 36, combine with 38
    # buy_36_ma_offset = 0.98
    # buy_36_ewo_max = -5.0
    # buy_36_cti_20_max = -0.82
    # buy_36_r_14_max = -97.0
    # buy_36_crsi_1h_min = 12.0

# new 35
    # buy_35_ma_offset = 0.98
    # buy_35_ewo_min = 1.1
    # buy_35_ewo_max = 8.2
    # buy_35_rsi_14_min = 26.0
    # buy_35_rsi_14_max = 40.0
    # buy_35_rsi_14_1h_max = 72.0
    # buy_35_crsi_1h_min = 12.0
    # buy_35_crsi_1h_max = 56.0
    # buy_35_cmf_42_min = -0.28
    # buy_35_chop_84_1h_max = 45.0
    # buy_35_r_14_max = -97.0

    buy_37_ma_offset = 0.984
    buy_37_ewo_min = 8.3
    buy_37_ewo_max = 11.1
    buy_37_rsi_14_min = 26.0
    buy_37_rsi_14_max = 46.0
    buy_37_crsi_1h_min = 12.0
    buy_37_crsi_1h_max = 56.0
    buy_37_cti_20_max = -0.85
    buy_37_cti_20_1h_max = 0.92
    buy_37_r_14_max = -97.0
    buy_37_close_1h_max = 0.1

    buy_38_ma_offset = 0.98
    buy_38_ewo_min = -7.8
    buy_38_ewo_max = -5.0
    buy_38_rsi_14_min = 26.0
    buy_38_rsi_14_max = 37.0
    buy_38_crsi_1h_min = 12.0
    buy_38_crsi_1h_max = 54.0
    buy_38_cti_20_max = -0.82
    buy_38_r_14_max = -97.0

    buy_39_cti_20_max = -0.1
    buy_39_r_1h_max = -22.0
    buy_39_cti_20_1h_min = -0.1
    buy_39_cti_20_1h_max = 0.4

    buy_40_cci_max = -150.0
    buy_40_rsi_max = 30.0
    buy_40_r_14_max = -99.9
    buy_40_cti_20_max = -0.8

    buy_41_ma_offset_high = 0.95
    buy_41_cti_20_max = -0.95
    buy_41_cci_max = -178.0
    buy_41_cci_osc_42_max = -0.65
    buy_41_ewo_1h_min = 0.5
    buy_41_r_480_1h_max = -14.0
    buy_41_crsi_1h_min = 14.0

    buy_42_ema_open_mult = 0.018
    buy_42_bb_offset = 0.992
    buy_42_ewo_1h_min = 2.8
    buy_42_cti_20_1h_min = -0.5
    buy_42_cti_20_1h_max = 0.88
    buy_42_r_480_1h_max = -12.0

    buy_43_bb40_bbdelta_close = 0.045
    buy_43_bb40_closedelta_close = 0.02
    buy_43_bb40_tail_bbdelta = 0.5
    buy_43_cti_20_max = -0.75
    buy_43_r_480_min = -94.0
    buy_43_cti_20_1h_min = -0.75
    buy_43_cti_20_1h_max = 0.45
    buy_43_r_480_1h_min = -80.0

    buy_44_ma_offset = 0.982
    buy_44_ewo_max = -18.0
    buy_44_cti_20_max = -0.73
    buy_44_crsi_1h_min = 8.0

    buy_45_bb40_bbdelta_close = 0.039
    buy_45_bb40_closedelta_close = 0.0231
    buy_45_bb40_tail_bbdelta = 0.24
    buy_45_ma_offset = 0.948
    buy_45_ewo_min = 2.0
    buy_45_ewo_1h_min = 2.0
    buy_45_cti_20_1h_max = 0.76
    buy_45_r_480_1h_max = -20.0

    buy_46_ema_open_mult = 0.0332
    buy_46_ewo_1h_min = 0.5
    buy_46_cti_20_1h_min = -0.9
    buy_46_cti_20_1h_max = 0.5

    buy_47_ewo_min = 3.2
    buy_47_ma_offset = 0.952
    buy_47_rsi_14_max = 46.0
    buy_47_cti_20_max = -0.93
    buy_47_r_14_max = -97.0
    buy_47_ewo_1h_min = 2.0
    buy_47_cti_20_1h_min = -0.9
    buy_47_cti_20_1h_max = 0.3

    buy_48_ewo_min = 8.5
    buy_48_ewo_1h_min = 14.0
    buy_48_r_480_min = -25.0
    buy_48_r_480_1h_min = -50.0
    buy_48_r_480_1h_max = -10.0
    buy_48_cti_20_1h_min = 0.5
    buy_48_crsi_1h_min = 10.0

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def sell_stoploss(self, current_profit: float, max_profit: float, max_loss: float, last_candle, previous_candle_1, trade: 'Trade', current_time: 'datetime') -> tuple:
        # Under & near EMA200, local uptrend move
        if (
            current_profit < -0.05
            and last_candle['close'] < last_candle['ema_200']
            and last_candle['cmf'] < 0.0
            and (last_candle['ema_200'] - last_candle['close']) / last_candle['close'] < 0.004
            and last_candle['rsi_14'] > previous_candle_1['rsi_14']
            and last_candle['rsi_14'] > (last_candle['rsi_14_1h'] + 10.0)
            and last_candle['sma_200_dec_24']
            and current_time - timedelta(minutes=2880) > trade.open_date_utc
        ):
            return True, 'signal_stoploss_u_e_1'

        # Under EMA200, local strong uptrend move
        if (
            current_profit < -0.08
            and last_candle['close'] < last_candle['ema_200']
            and last_candle['cmf'] < 0.0
            and last_candle['rsi_14'] > previous_candle_1['rsi_14']
            and last_candle['rsi_14'] > last_candle['rsi_14_1h'] + 24.0
            and last_candle['sma_200_dec_20']
            and last_candle['sma_200_dec_24']
            and current_time - timedelta(minutes=2880) > trade.open_date_utc
        ):
            return True, 'signal_stoploss_u_e_2'

        # Under EMA200, pair negative, low max rate
        if (
            current_profit < -0.08
            and max_profit < 0.04
            and last_candle['close'] < last_candle['ema_200']
            and last_candle['ema_25'] < last_candle['ema_50']
            and last_candle['sma_200_dec_20']
            and last_candle['sma_200_dec_24']
            and last_candle['sma_200_dec_20_1h']
            and last_candle['ema_vwma_osc_32'] < 0.0
            and last_candle['ema_vwma_osc_64'] < 0.0
            and last_candle['ema_vwma_osc_96'] < 0.0
            and last_candle['cmf'] < -0.0
            and last_candle['cmf_1h'] < -0.0
            and last_candle['close'] < last_candle['sup_level_1h']
            and not last_candle['btc_not_downtrend_1h']
            and current_time - timedelta(minutes=1440) > trade.open_date_utc
        ):
            return True, 'signal_stoploss_u_e_doom'

        # Under EMA200, pair and BTC negative, low max rate
        if (
            -0.05 > current_profit > -0.09
            and not last_candle['btc_not_downtrend_1h']
            and last_candle['ema_vwma_osc_32'] < 0.0
            and last_candle['ema_vwma_osc_64'] < 0.0
            and max_profit < 0.005
            and max_loss < 0.09
            and last_candle['sma_200_dec_24']
            and last_candle['cmf'] < -0.0
            and last_candle['close'] < last_candle['ema_200']
            and last_candle['ema_25'] < last_candle['ema_50']
            and last_candle['cti_20'] < -0.8
            and last_candle['r_480'] < -50.0
        ):
            return True, 'sell_stoploss_u_e_b_1'

        # Under EMA200, pair and BTC negative, CTI, downtrend, normal max rate
        if (
            -0.1 > current_profit > -0.2
            and not last_candle['btc_not_downtrend_1h']
            and last_candle['ema_vwma_osc_32'] < 0.0
            and last_candle['ema_vwma_osc_64'] < 0.0
            and last_candle['ema_vwma_osc_96'] < 0.0
            and max_profit < 0.05
            and max_loss < 0.2
            and last_candle['sma_200_dec_24']
            and last_candle['sma_200_dec_20_1h']
            and last_candle['cmf'] < -0.45
            and last_candle['close'] < last_candle['ema_200']
            and last_candle['ema_25'] < last_candle['ema_50']
            and last_candle['cti_20'] < -0.8
            and last_candle['r_480'] < -97.0
        ):
            return True, 'sell_stoploss_u_e_b_2'

        return False, None

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

        # Stoplosses
        # if current_profit < -0.01:
        #     sell, signal_name = self.sell_stoploss(current_profit, max_profit, max_loss, candle_1, candle_2, trade, current_time)
        #     if sell:
        #         return f"{signal_name} ( {buy_tag})"

        # do nothing with small losses or profits
        if not (-0.02 <= current_profit <= 0.02):
            sell, signal_name = self.sell_multiple_indicators(candle_1, candle_2)
            if sell:
                return f"{signal_name} ( {buy_tag})"

        return None

    def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

    def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

    def range_maxgap(self, dataframe: DataFrame, length: int) -> float:
        """
        Maximum Price Gap across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        return dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()

    def range_maxgap_adjusted(self, dataframe: DataFrame, length: int, adjustment: float) -> float:
        """
        Maximum Price Gap across interval adjusted.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        :param adjustment: int The adjustment to be applied
        """
        return self.range_maxgap(dataframe, length) / adjustment

    def range_height(self, dataframe: DataFrame, length: int) -> float:
        """
        Current close distance to range bottom.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        return dataframe['close'] - dataframe['close'].rolling(length).min()

    def safe_pump(self, dataframe: DataFrame, length: int, thresh: float, pull_thresh: float) -> bool:
        """
        Determine if entry after a pump is safe.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        :param thresh: int Maximum percentage change threshold
        :param pull_thresh: int Pullback from interval maximum threshold
        """
        return (dataframe[f'oc_pct_change_{length}'] < thresh) | (self.range_maxgap_adjusted(dataframe, length, pull_thresh) > self.range_height(dataframe, length))

    def safe_dips(self, dataframe: DataFrame, thresh_0, thresh_2, thresh_12, thresh_144) -> bool:
        """
        Determine if dip is safe to enter.

        :param dataframe: DataFrame The original OHLC dataframe
        :param thresh_0: Threshold value for 0 length top pct change
        :param thresh_2: Threshold value for 2 length top pct change
        :param thresh_12: Threshold value for 12 length top pct change
        :param thresh_144: Threshold value for 144 length top pct change
        """
        return ((dataframe['tpct_change_0'] < thresh_0) &
                (dataframe['tpct_change_2'] < thresh_2) &
                (dataframe['tpct_change_12'] < thresh_12) &
                (dataframe['tpct_change_144'] < thresh_144))

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.info_timeframe_1h) for pair in pairs]
        informative_pairs.extend([(pair, self.info_timeframe_1d) for pair in pairs])

        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.info_timeframe_1h))
        informative_pairs.append((btc_info_pair, self.info_timeframe_1d))

        return informative_pairs

    def informative_1d_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1d: DataFrame = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1d)

        # Top traded coins
        if self.coin_metrics['top_traded_enabled']:
            informative_1d = informative_1d.merge(self.coin_metrics['tt_dataframe'], on='date', how='left')
            informative_1d['is_top_traded'] = informative_1d.apply(lambda row: self.is_top_coin(metadata['pair'], row, self.coin_metrics['top_traded_len']), axis=1)
            column_names = [f"Coin #{i}" for i in range(1, self.coin_metrics['top_traded_len'] + 1)]
            informative_1d.drop(columns = column_names, inplace=True)
        # Top grossing coins
        if self.coin_metrics['top_grossing_enabled']:
            informative_1d = informative_1d.merge(self.coin_metrics['tg_dataframe'], on='date', how='left')
            informative_1d['is_top_grossing'] = informative_1d.apply(lambda row: self.is_top_coin(metadata['pair'], row, self.coin_metrics['top_grossing_len']), axis=1)
            column_names = [f"Coin #{i}" for i in range(1, self.coin_metrics['top_grossing_len'] + 1)]
            informative_1d.drop(columns = column_names, inplace=True)

        # Pivots
        informative_1d['pivot'], informative_1d['res1'], informative_1d['res2'], informative_1d['res3'], informative_1d['sup1'], informative_1d['sup2'], informative_1d['sup3'] = pivot_points(informative_1d, mode='fibonacci')

        # Smoothed Heikin-Ashi
        # informative_1d['open_sha'], informative_1d['close_sha'], informative_1d['low_sha'] = HeikinAshi(informative_1d, smooth_inputs=True, smooth_outputs=False, length=10)

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

        # EMA
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_15'] = ta.EMA(informative_1h, timeperiod=15)
        informative_1h['ema_20'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_25'] = ta.EMA(informative_1h, timeperiod=25)
        informative_1h['ema_26'] = ta.EMA(informative_1h, timeperiod=26)
        informative_1h['ema_35'] = ta.EMA(informative_1h, timeperiod=35)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h['sma_200_dec_20'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)

        # RSI
        informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)
        
        # Chopiness
        informative_1h['chop_84'] = qtpylib.chopiness(dataframe, 84)

        # EWO
        informative_1h['ewo'] = ewo(informative_1h, 50, 200)

        # S/R
        # res_series = informative_1h['high'].rolling(window=5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
        sup_series = informative_1h['low'].rolling(window=5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
        # informative_1h['res_level'] = Series(np.where(res_series, np.where(informative_1h['close'] > informative_1h['open'], informative_1h['close'], informative_1h['open']), float('NaN'))).ffill()
        # informative_1h['res_hlevel'] = Series(np.where(res_series, informative_1h['high'], float('NaN'))).ffill()
        informative_1h['sup_level'] = Series(np.where(sup_series, np.where(informative_1h['close'] < informative_1h['open'], informative_1h['close'], informative_1h['open']), float('NaN'))).ffill()

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb20_2_low'] = bollinger['lower']
        informative_1h['bb20_2_mid'] = bollinger['mid']
        informative_1h['bb20_2_upp'] = bollinger['upper']

        boll_close = qtpylib.bollinger_bands(informative_1h['close'], window=18, stds=2)
        informative_1h['bb18_close_upp'] = boll_close['upper']

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

        # Ichimoku
        # ichi = ichimoku(informative_1h, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        # informative_1h['chikou_span'] = ichi['chikou_span']
        # informative_1h['tenkan_sen'] = ichi['tenkan_sen']
        # informative_1h['kijun_sen'] = ichi['kijun_sen']
        # informative_1h['senkou_a'] = ichi['senkou_span_a']
        # informative_1h['senkou_b'] = ichi['senkou_span_b']
        # informative_1h['leading_senkou_span_a'] = ichi['leading_senkou_span_a']
        # informative_1h['leading_senkou_span_b'] = ichi['leading_senkou_span_b']
        # informative_1h['chikou_span_greater'] = (informative_1h['chikou_span'] > informative_1h['senkou_a']).shift(30).fillna(False)
        # informative_1h.loc[:, 'cloud_top'] = informative_1h.loc[:, ['senkou_a', 'senkou_b']].max(axis=1)

        # SSL
        # ssl_down, ssl_up = SSLChannels(informative_1h, 10)
        # informative_1h['ssl_down'] = ssl_down
        # informative_1h['ssl_up'] = ssl_up

        # MOMDIV
        mom = momdiv(informative_1h)
        informative_1h['momdiv_buy'] = mom['momdiv_buy']
        # informative_1h['momdiv_sell'] = mom['momdiv_sell']
        # informative_1h['momdiv_coh'] = mom['momdiv_coh']
        # informative_1h['momdiv_col'] = mom['momdiv_col']        

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low'] = bb_40_std2['lower']
        dataframe['bb40_2_mid'] = bb_40_std2['mid']
        dataframe['bb40_2_delta'] = (bb_40_std2['mid'] - dataframe['bb40_2_low']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['bb40_2_low']).abs()

        # BB 20 - STD2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        # EMA
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_15'] = ta.EMA(dataframe, timeperiod=15)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_200_dec_24'] = dataframe['sma_200'] < dataframe['sma_200'].shift(24)

        # MFI
        dataframe['mfi_14'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['mfi_42'] = ta.MFI(dataframe, timeperiod=42)

        # CMF
        dataframe['cmf_14'] = chaikin_money_flow(dataframe, 14)
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        dataframe['cmf_42'] = chaikin_money_flow(dataframe, 42)

        # EWO
        dataframe['ewo'] = ewo(dataframe, 50, 200)
        dataframe['ewo_25'] = ewo(dataframe, 5, 25)

        # RSI
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_42'] = ta.RSI(dataframe, timeperiod=42)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        # Zero-Lag EMA
        dataframe['zlema_61'] = zlema(dataframe, 61)
        dataframe['zlema_68'] = zlema(dataframe, 68)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_42'] = williams_r(dataframe, period=42)
        dataframe['r_480'] = williams_r(dataframe, period=480)

        # Stochastic RSI
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=96, fastk_period=3, fastd_period=3, fastd_matype=0)
        dataframe['stochrsi_fastk_96'] = stochrsi['fastk']
        dataframe['stochrsi_fastd_96'] = stochrsi['fastd']

        # EMA of VWMA Oscillator
        dataframe['ema_vwma_osc_32'] = ema_vwma_osc(dataframe, 32)
        dataframe['ema_vwma_osc_64'] = ema_vwma_osc(dataframe, 64)
        dataframe['ema_vwma_osc_96'] = ema_vwma_osc(dataframe, 96)

        # hull
        dataframe['hull_75'] = hull(dataframe, 75)

        # CRSI
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi_100'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3
        dataframe['crsi_480'] = (ta.RSI(dataframe['close'], timeperiod=6) + ta.RSI(crsi_updown, timeperiod=3) + ta.ROC(dataframe['close'], 480)) / 3

        # CTI
        dataframe['cti_14'] = pta.cti(dataframe['close'], length=14)
        dataframe['cti_20'] = pta.cti(dataframe['close'], length=20)
        dataframe['cti_42'] = pta.cti(dataframe['close'], length=42)

        # For sell checks
        dataframe['crossed_below_ema_12_26'] = qtpylib.crossed_below(dataframe['ema_12'], dataframe['ema_26'])

        dataframe['sma_21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma_68'] = ta.SMA(dataframe, timeperiod=68)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        # CCI
        dataframe['cci'] = ta.CCI(dataframe, source='hlc3', timeperiod=20)

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

        # MOMDIV
        mom = momdiv(dataframe)
        dataframe['momdiv_buy'] = mom['momdiv_buy']
        # dataframe['momdiv_sell'] = mom['momdiv_sell']
        # dataframe['momdiv_coh'] = mom['momdiv_coh']
        # dataframe['momdiv_col'] = mom['momdiv_col']

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_30'] = dataframe['volume'].rolling(30).mean()
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

    def resampled_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] base_tf_btc_indicators took: {tok - tik:0.4f} seconds.")

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['not_downtrend'] = ((dataframe['close'] > dataframe['close'].shift(2)) | (dataframe['rsi_14'] > 50))

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] info_tf_btc_indicators took: {tok - tik:0.4f} seconds.")

        return dataframe

    def daily_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['pivot'], dataframe['res1'], dataframe['res2'], dataframe['res3'], dataframe['sup1'], dataframe['sup2'], dataframe['sup3'] = pivot_points(dataframe, mode='fibonacci')

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] daily_tf_btc_indicators took: {tok - tik:0.4f} seconds.")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        '''
        --> BTC informative (5m/1h)
        ___________________________________________________________________________________________
        '''
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        if self.has_BTC_daily_tf:
            btc_daily_tf = self.dp.get_pair_dataframe(btc_info_pair, '1d')
            btc_daily_tf = self.daily_tf_btc_indicators(btc_daily_tf, metadata)
            dataframe = merge_informative_pair(dataframe, btc_daily_tf, self.timeframe, '1d', ffill=True)
            drop_columns = [f"{s}_1d" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        if self.has_BTC_info_tf:
            btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.info_timeframe_1h)
            btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
            dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.info_timeframe_1h, ffill=True)
            drop_columns = [f"{s}_{self.info_timeframe_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        if self.has_BTC_base_tf:
            btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
            btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
            dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
            drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> Informative timeframe
        ___________________________________________________________________________________________
        '''
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


        '''
        --> Resampled to another timeframe
        ___________________________________________________________________________________________
        '''
        if self.res_timeframe != 'none':
            resampled = resample_to_interval(dataframe, timeframe_to_minutes(self.res_timeframe))
            resampled = self.resampled_tf_indicators(resampled, metadata)
            # Merge resampled info dataframe
            dataframe = resampled_merge(dataframe, resampled, fill_na=True)
            dataframe.rename(columns=lambda s: f"{s}_{self.res_timeframe}" if "resample_" in s else s, inplace=True)
            dataframe.rename(columns=lambda s: s.replace("resample_{}_".format(self.res_timeframe.replace("m","")), ""), inplace=True)
            drop_columns = [f"{s}_{self.res_timeframe}" for s in ['date']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> The indicators for the normal (5m) timeframe
        ___________________________________________________________________________________________
        '''
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] Populate indicators took a total of: {tok - tik:0.4f} seconds.")

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        for index in self.buy_protection_params:
            item_buy_protection_list = [dataframe['volume'] >= self.min_vol_candle]
            item_buy_protection_list.append(dataframe['volume_12'] >= self.min_vol_1h)
            item_buy_protection_list.append(dataframe['pump'] < 1.500)
            item_buy_protection_list.append(dataframe['dump'] > 0.667)
            global_buy_protection_params = self.buy_protection_params[index]

            if self.buy_params[f"buy_condition_{index}_enable"]:
                # Standard protections - Common to every condition
                # -----------------------------------------------------------------------------------------
                if global_buy_protection_params["ema_fast"]:
                    item_buy_protection_list.append(dataframe[f"ema_{global_buy_protection_params['ema_fast_len']}"] > dataframe['ema_200'])
                if global_buy_protection_params["ema_slow"]:
                    item_buy_protection_list.append(dataframe[f"ema_{global_buy_protection_params['ema_slow_len']}_1h"] > dataframe['ema_200_1h'])
                if global_buy_protection_params["close_above_ema_fast"]:
                    item_buy_protection_list.append(dataframe['close'] > dataframe[f"ema_{global_buy_protection_params['close_above_ema_fast_len']}"])
                if global_buy_protection_params["close_above_ema_slow"]:
                    item_buy_protection_list.append(dataframe['close'] > dataframe[f"ema_{global_buy_protection_params['close_above_ema_slow_len']}_1h"])
                if global_buy_protection_params["sma200_rising"]:
                    item_buy_protection_list.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(global_buy_protection_params['sma200_rising_val'])))
                if global_buy_protection_params["sma200_1h_rising"]:
                    item_buy_protection_list.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(global_buy_protection_params['sma200_1h_rising_val'])))
                # if global_buy_protection_params["safe_dips_threshold_0"] is not None:
                #     item_buy_protection_list.append(dataframe['tpct_change_0'] < global_buy_protection_params["safe_dips_threshold_0"])
                # if global_buy_protection_params["safe_dips_threshold_2"] is not None:
                #     item_buy_protection_list.append(dataframe['tpct_change_2'] < global_buy_protection_params["safe_dips_threshold_2"])
                # if global_buy_protection_params["safe_dips_threshold_12"] is not None:
                #     item_buy_protection_list.append(dataframe['tpct_change_12'] < global_buy_protection_params["safe_dips_threshold_12"])
                # if global_buy_protection_params["safe_dips_threshold_144"] is not None:
                #     item_buy_protection_list.append(dataframe['tpct_change_144'] < global_buy_protection_params["safe_dips_threshold_144"])
                # if global_buy_protection_params["safe_pump"]:
                #     item_buy_protection_list.append(dataframe[f"safe_pump_{global_buy_protection_params['safe_pump_period']}_{global_buy_protection_params['safe_pump_type']}_1h"])
                if global_buy_protection_params['btc_1h_not_downtrend']:
                    item_buy_protection_list.append(dataframe['btc_not_downtrend_1h'])
                # if global_buy_protection_params['close_over_pivot_type'] != 'none':
                #     item_buy_protection_list.append(dataframe['close'] > dataframe[f"{global_buy_protection_params['close_over_pivot_type']}_1d"] * global_buy_protection_params['close_over_pivot_offset'])
                # if global_buy_protection_params['close_under_pivot_type'] != 'none':
                #     item_buy_protection_list.append(dataframe['close'] < dataframe[f"{global_buy_protection_params['close_under_pivot_type']}_1d"] * global_buy_protection_params['close_under_pivot_offset'])
                if self.config['runmode'].value not in ('live', 'dry_run'):
                    if self.has_bt_agefilter:
                        item_buy_protection_list.append(dataframe['bt_agefilter_1d'] >= self.bt_min_age_days)
                else:
                    if self.has_downtime_protection:
                        item_buy_protection_list.append(dataframe['live_data_ok'])

                # Buy conditions
                # -----------------------------------------------------------------------------------------
                item_buy_logic = []
                item_buy_logic.append(reduce(lambda x, y: x & y, item_buy_protection_list))

                # Condition #1
                if index == 1:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(((dataframe['close'] - dataframe['open'].rolling(12).min()) / dataframe['open'].rolling(12).min()) > self.buy_1_min_inc)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_1_rsi_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_1_r_14_max)
                    item_buy_logic.append(dataframe['mfi_14'] < self.buy_1_mfi_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] > self.buy_1_rsi_1h_min)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_1_rsi_1h_max)

                # Condition #2
                elif index == 2:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['rsi_14'] < dataframe['rsi_14_1h'] - self.buy_2_rsi_1h_diff)
                    item_buy_logic.append(dataframe['mfi_14'] < self.buy_2_mfi)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_2_cti_20_max)
                    item_buy_logic.append(dataframe['r_480'] > self.buy_2_r_480_min)
                    item_buy_logic.append(dataframe['r_480'] < self.buy_2_r_480_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_2_cti_20_1h_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_2_volume))

                # Condition #3
                elif index == 3:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['bb40_2_low'].shift().gt(0))
                    item_buy_logic.append(dataframe['bb40_2_delta'].gt(dataframe['close'] * self.buy_3_bb40_bbdelta_close))
                    item_buy_logic.append(dataframe['closedelta'].gt(dataframe['close'] * self.buy_3_bb40_closedelta_close))
                    item_buy_logic.append(dataframe['tail'].lt(dataframe['bb40_2_delta'] * self.buy_3_bb40_tail_bbdelta))
                    item_buy_logic.append(dataframe['close'].lt(dataframe['bb40_2_low'].shift()))
                    item_buy_logic.append(dataframe['close'].le(dataframe['close'].shift()))
                    item_buy_logic.append(dataframe['cci_osc_42'] > self.buy_3_cci_osc_42_min)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_3_crsi_1h_min)
                    item_buy_logic.append(dataframe['r_480_1h'] > self.buy_3_r_480_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_3_cti_20_1h_max)

                # Condition #4
                elif index == 4:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_50'])
                    item_buy_logic.append(dataframe['close'] < self.buy_4_bb20_close_bblowerband * dataframe['bb20_2_low'])
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_30'].shift(1) * self.buy_4_bb20_volume))
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_4_cti_20_max)

                # Condition #5
                elif index == 5:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] > (dataframe['ema_200_1h'] * self.buy_5_ema_rel))
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_5_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_5_bb_offset))
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_5_cti_20_max)
                    item_buy_logic.append(dataframe['rsi_14'] > self.buy_5_rsi_14_min)
                    item_buy_logic.append(dataframe['mfi_14'] > self.buy_5_mfi_min)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_5_r_14_max)
                    item_buy_logic.append(dataframe['r_14'].shift(1) < self.buy_5_r_14_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_5_crsi_1h_min)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_5_volume))

                # Condition #6
                elif index == 6:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_6_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_6_bb_offset))
                    item_buy_logic.append(dataframe['r_14'] < self.buy_6_r_14_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_6_cti_20_1h_min)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_6_crsi_1h_min)

                # Condition #7
                elif index == 7:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_7_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_7_ma_offset)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_7_cti_20_max)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_7_rsi_max)

                # Condition #8
                elif index == 8:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_20'] > dataframe['ema_50'])
                    item_buy_logic.append(dataframe['ema_15'] > dataframe['ema_100'])
                    item_buy_logic.append(dataframe['ema_200'] > dataframe['sma_200'])

                    # Logic
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_8_bb_offset))
                    item_buy_logic.append(dataframe['r_14'] < self.buy_8_r_14_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_8_cti_20_1h_max)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_8_r_480_1h_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_8_volume))

                # Condition #9
                elif index == 9:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_50'] > dataframe['ema_200'])

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_9_ma_offset)
                    item_buy_logic.append(dataframe['close'] < dataframe['bb20_2_low'] * self.buy_9_bb_offset)
                    item_buy_logic.append(dataframe['mfi_14'] < self.buy_9_mfi_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_9_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_9_r_14_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] > self.buy_9_rsi_1h_min)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_9_rsi_1h_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_9_crsi_1h_min)

                # Condition #10
                elif index == 10:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_50_1h'] > dataframe['ema_100_1h'])

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_10_ma_offset_high)
                    item_buy_logic.append(dataframe['close'] < dataframe['bb20_2_low'] * self.buy_10_bb_offset)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_10_r_14_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_10_cti_20_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_10_cti_20_1h_max)

                # Condition #11
                elif index == 11:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(((dataframe['close'] - dataframe['open'].rolling(6).min()) / dataframe['open'].rolling(6).min()) > self.buy_11_min_inc)
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_11_ma_offset)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_11_rsi_max)
                    item_buy_logic.append(dataframe['mfi_14'] < self.buy_11_mfi_max)
                    item_buy_logic.append(dataframe['cci'] < self.buy_11_cci_max)
                    # item_buy_logic.append(dataframe['cci_osc_42'] < self.buy_11_cci_osc_42_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] > self.buy_11_rsi_1h_min)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_11_rsi_1h_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_11_cti_20_1h_max)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_11_r_480_1h_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_11_crsi_1h_min)

                # Condition #12
                elif index == 12:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_12_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_12_ewo_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_12_rsi_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_12_cti_20_max)

                # Condition #13
                elif index == 13:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_50_1h'] > dataframe['ema_100_1h'])

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_13_ma_offset)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_13_cti_20_max)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_13_ewo_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_13_cti_20_1h_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_13_crsi_1h_min)

                # Condition #14
                elif index == 14:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_14_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_14_bb_offset))
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_14_ma_offset)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_14_cti_20_max)

                # Condition #15
                elif index == 15:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_15_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_15_rsi_min)
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_15_ma_offset)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_15_cti_20_1h_min)

                # Condition #16
                elif index == 16:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_16_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_16_ewo_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_16_rsi_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_16_cti_20_max)

                # Condition #17
                elif index == 17:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_17_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_17_ewo_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_17_cti_20_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_17_crsi_1h_min)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_17_volume))

                # Condition #18
                elif index == 18:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['sma_200'] > dataframe['sma_200'].shift(20))
                    item_buy_logic.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(36))

                    # Logic
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_18_bb_offset))
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_18_rsi_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_18_cti_20_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_18_cti_20_1h_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_18_volume))

                # Condition #19
                elif index == 19:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_vwma_osc_32'] > 0)
                    item_buy_logic.append(dataframe['ema_vwma_osc_64'] > 0)
                    item_buy_logic.append(dataframe['ema_vwma_osc_96'] > 0)

                    # Logic
                    item_buy_logic.append(dataframe['close'].shift(1) > dataframe['ema_100_1h'])
                    item_buy_logic.append(dataframe['low'] < dataframe['ema_100_1h'])
                    item_buy_logic.append(dataframe['close'] > dataframe['ema_100_1h'])
                    item_buy_logic.append(dataframe['chop'] < self.buy_19_chop_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] > self.buy_19_rsi_1h_min)

                # Condition #20
                elif index == 20:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_20_rsi_14_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_20_rsi_14_1h_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_20_cti_20_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_20_volume))

                # Condition #21
                elif index == 21:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_21_rsi_14_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_21_rsi_14_1h_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_21_cti_20_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_21_volume))

                # Condition #22
                elif index == 22:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_100_1h'] > dataframe['ema_100_1h'].shift(12))
                    item_buy_logic.append(dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(36))

                    # Logic
                    item_buy_logic.append((dataframe['volume_mean_4'] * self.buy_22_volume) > dataframe['volume'])
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_22_ma_offset)
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_22_bb_offset))
                    item_buy_logic.append(dataframe['ewo'] > self.buy_22_ewo_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_22_rsi_14_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_22_cti_20_max)
                    item_buy_logic.append(dataframe['r_480'] < self.buy_22_r_480_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_22_cti_20_1h_min)

                # Condition #23
                elif index == 23:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_23_bb_offset))
                    item_buy_logic.append(dataframe['ewo'] > self.buy_23_ewo_min)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_23_cti_20_max)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_23_rsi_14_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_23_rsi_14_1h_max)
                    item_buy_logic.append(dataframe['r_480_1h'] > self.buy_23_r_480_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < 0.92)

                # Condition #24
                elif index == 24:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_12_1h'].shift(12) < dataframe['ema_35_1h'].shift(12))
                    item_buy_logic.append(dataframe['ema_12_1h'] > dataframe['ema_35_1h'])
                    item_buy_logic.append(dataframe['cmf_1h'].shift(12) < 0)
                    item_buy_logic.append(dataframe['cmf_1h'] > 0)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_24_rsi_14_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] > self.buy_24_rsi_14_1h_min)

                # Condition #25
                elif index == 25:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['rsi_20'] < dataframe['rsi_20'].shift())
                    item_buy_logic.append(dataframe['rsi_4'] < self.buy_25_rsi_4_max)
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_15'] * self.buy_25_ma_offset)
                    item_buy_logic.append(dataframe['ema_20_1h'] > dataframe['ema_26_1h'])
                    item_buy_logic.append(dataframe['open'] > (dataframe['sma_15'] * self.buy_25_ma_offset))
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_25_cti_20_max)
                    item_buy_logic.append(dataframe['cci'] < self.buy_25_cci_max)
                    # item_buy_logic.append(dataframe['cci_osc_42'] < self.buy_25_cci_osc_42_max)

                # Condition #26
                elif index == 26:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['zlema_61'] * self.buy_26_zema_low_offset)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_26_cti_20_max)
                    item_buy_logic.append(dataframe['cci'] < self.buy_26_cci_max)
                    # item_buy_logic.append(dataframe['cci_osc_42'] < self.buy_26_cci_osc_42_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_26_r_14_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_26_cti_20_1h_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_26_volume))

                # Condition #27
                elif index == 27:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['r_480'] < self.buy_27_wr_max)
                    item_buy_logic.append(dataframe['r_14'] == self.buy_27_r_14)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_27_wr_1h_max)
                    item_buy_logic.append(dataframe['rsi_14_1h'] + dataframe['rsi_14'] < self.buy_27_rsi_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_27_volume))

                # Condition #28
                elif index == 28:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_vwma_osc_64'] > 0)
                    item_buy_logic.append(dataframe['close'] < dataframe['hull_75'] * self.buy_28_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_28_ewo_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_28_rsi_14_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_28_cti_20_max)
                    item_buy_logic.append(dataframe['cti_20'].shift(1) < self.buy_28_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_28_r_14_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_28_cti_20_1h_max)

                # Condition #29
                elif index == 29:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_vwma_osc_64'] > 0)
                    item_buy_logic.append(dataframe['close'] < dataframe['hull_75'] * self.buy_29_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_29_ewo_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_29_cti_20_max)

                # Condition #30
                elif index == 30:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_vwma_osc_64'] < 0)
                    item_buy_logic.append(dataframe['close'] < dataframe['zlema_68'] * self.buy_30_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_30_ewo_min)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_30_ewo_max)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_30_rsi_14_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_30_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_30_r_14_max)

                # Condition #31
                elif index == 31:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_vwma_osc_64'] < 0)
                    item_buy_logic.append(dataframe['close'] < dataframe['zlema_68'] * self.buy_31_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_31_ewo_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_31_r_14_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_31_cti_20_max)

                # Condition #32 - Quick mode buy
                elif index == 32:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_20_1h'] > dataframe['ema_25_1h'])
                    item_buy_logic.append(dataframe['rsi_20'] < dataframe['rsi_20'].shift(1))
                    item_buy_logic.append(dataframe['rsi_4'] < self.buy_32_rsi_4_max)
                    item_buy_logic.append(dataframe['rsi_14'] > self.buy_32_rsi_14_min)
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_15'] * self.buy_32_ma_offset)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_32_cti_20_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_32_crsi_1h_min)
                    item_buy_logic.append(dataframe['crsi_1h'] < self.buy_32_crsi_1h_max)

                # Condition #33 - Quick mode buy
                elif index == 33:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < (dataframe['ema_13'] * self.buy_33_ma_offset))
                    item_buy_logic.append(dataframe['ewo'] > self.buy_33_ewo_min)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_33_cti_20_max)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_33_rsi_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_33_r_14_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_33_cti_20_1h_max)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_33_volume))

                # Condition #34 - Quick mode buy
                elif index == 34:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['ema_13'] * self.buy_34_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_34_ewo_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_34_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_34_r_14_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_34_crsi_1h_min)
                    item_buy_logic.append(dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_34_volume))

                # Condition #35
                elif index == 35:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_75'] * self.buy_35_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_35_ewo_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_35_rsi_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_35_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_35_r_14_max)

                # Condition #36
                elif index == 36:
                    # Non-Standard protections (add below)

                    # Logic
                    # item_buy_logic.append(dataframe['pm'] <= dataframe['pmax_thresh'])
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_75'] * self.buy_36_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_36_ewo_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_36_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_36_r_14_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_36_crsi_1h_min)

                    # item_buy_logic.append(dataframe['close'] < dataframe['sma_75'] * self.buy_37_ma_offset)
                    # item_buy_logic.append(dataframe['ewo'] > self.buy_37_ewo_min)
                    # item_buy_logic.append(dataframe['ewo'] < self.buy_37_ewo_max)
                    # item_buy_logic.append(dataframe['rsi_14'] > self.buy_37_rsi_14_min)
                    # item_buy_logic.append(dataframe['rsi_14'] < self.buy_37_rsi_14_max)
                    # item_buy_logic.append(dataframe['rsi_14_1h'] < self.buy_37_rsi_14_1h_max)
                    # item_buy_logic.append(dataframe['crsi_1h'] > self.buy_37_crsi_1h_min)
                    # item_buy_logic.append(dataframe['crsi_1h'] < self.buy_37_crsi_1h_max)
                    # item_buy_logic.append(dataframe['cmf_42'] > self.buy_37_cmf_42_min)
                    # item_buy_logic.append(dataframe['chop_84_1h'] < self.buy_37_chop_84_1h_max)
                    # item_buy_logic.append(dataframe['r_14'] < self.buy_37_r_14_max)

                # Condition #37
                elif index == 37:
                    # Non-Standard protections (add below)

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_75'] * self.buy_37_ma_offset)
                    item_buy_logic.append(((dataframe['close_1h'].shift(12) - dataframe['close_1h']) / dataframe['close_1h']) < self.buy_37_close_1h_max)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_37_ewo_min)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_37_ewo_max)
                    item_buy_logic.append(dataframe['rsi_14'] > self.buy_37_rsi_14_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_37_rsi_14_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_37_crsi_1h_min)
                    item_buy_logic.append(dataframe['crsi_1h'] < self.buy_37_crsi_1h_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_37_cti_20_max)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_37_cti_20_1h_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_37_r_14_max)

                # Condition #38
                elif index == 38:
                    # Non-Standard protections (add below)

                    # Logic
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_75'] * self.buy_38_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_38_ewo_min)
                    item_buy_logic.append(dataframe['ewo'] < self.buy_38_ewo_max)
                    item_buy_logic.append(dataframe['rsi_14'] > self.buy_38_rsi_14_min)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_38_rsi_14_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_38_crsi_1h_min)
                    item_buy_logic.append(dataframe['crsi_1h'] < self.buy_38_crsi_1h_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_38_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_38_r_14_max)

                # Condition #39 - Ichimoku
                # elif index == 39:
                #     # Non-Standard protections (add below)

                #     # Logic
                #     item_buy_logic.append(dataframe['tenkan_sen_1h'] > dataframe['kijun_sen_1h'])
                #     item_buy_logic.append(dataframe['close'] > dataframe['cloud_top_1h'])
                #     item_buy_logic.append(dataframe['leading_senkou_span_a_1h'] > dataframe['leading_senkou_span_b_1h'])
                #     item_buy_logic.append(dataframe['chikou_span_greater_1h'])
                #     item_buy_logic.append(dataframe['ssl_up_1h'] > dataframe['ssl_down_1h'])
                #     item_buy_logic.append(dataframe['close'] < dataframe['ssl_up_1h'])
                #     item_buy_logic.append(dataframe['rsi_14_1h'] > dataframe['rsi_14_1h'].shift(12))
                #     item_buy_logic.append(dataframe['cti_20'] < self.buy_39_cti_20_max)
                #     item_buy_logic.append(dataframe['r_480_1h'] < self.buy_39_r_1h_max)
                #     item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_39_cti_20_1h_min)
                #     item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_39_cti_20_1h_max)
                #     # Start of trend
                #     item_buy_logic.append(
                #         (dataframe['leading_senkou_span_a_1h'].shift(12) < dataframe['leading_senkou_span_b_1h'].shift(12))
                #     )

                # Condition #40
                elif index == 40:
                    # Non-Standard protections (add below)

                    # Logic
                    item_buy_logic.append(dataframe['momdiv_buy_1h'] == True)
                    item_buy_logic.append(dataframe['cci'] < self.buy_40_cci_max)
                    # item_buy_logic.append(dataframe['cci_osc_42'] < self.buy_40_cci_osc_42_max)
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_40_rsi_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_40_r_14_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_40_cti_20_max)

                # Condition #41
                elif index == 41:
                    # Non-Standard protections (add below)

                    # Logic
                    item_buy_logic.append(dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12))
                    item_buy_logic.append(dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24))
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_75'] * self.buy_41_ma_offset_high)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_41_cti_20_max)
                    item_buy_logic.append(dataframe['cci'] < self.buy_41_cci_max)
                    # item_buy_logic.append(dataframe['cci_osc_42'] < self.buy_41_cci_osc_42_max)
                    item_buy_logic.append(dataframe['ewo_1h'] > self.buy_41_ewo_1h_min)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_41_r_480_1h_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_41_crsi_1h_min)

                # Condition #42
                elif index == 42:
                    # Non-Standard protections (add below)

                    # Logic
                    item_buy_logic.append(dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12))
                    item_buy_logic.append(dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24))
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_42_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_42_bb_offset))
                    item_buy_logic.append(dataframe['ewo_1h'] > self.buy_42_ewo_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_42_cti_20_1h_min)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_42_r_480_1h_max)                    

                # Condition #43
                elif index == 43:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12))
                    item_buy_logic.append(dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24))
                    item_buy_logic.append(dataframe['bb40_2_low'].shift().gt(0))
                    item_buy_logic.append(dataframe['bb40_2_delta'].gt(dataframe['close'] * self.buy_43_bb40_bbdelta_close))
                    item_buy_logic.append(dataframe['closedelta'].gt(dataframe['close'] * self.buy_43_bb40_closedelta_close))
                    item_buy_logic.append(dataframe['tail'].lt(dataframe['bb40_2_delta'] * self.buy_43_bb40_tail_bbdelta))
                    item_buy_logic.append(dataframe['close'].lt(dataframe['bb40_2_low'].shift()))
                    item_buy_logic.append(dataframe['close'].le(dataframe['close'].shift()))
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_43_cti_20_max)
                    item_buy_logic.append(dataframe['r_480'] > self.buy_43_r_480_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_43_cti_20_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_43_cti_20_1h_max)
                    item_buy_logic.append(dataframe['r_480_1h'] > self.buy_43_r_480_1h_min)

                # Condition #44
                elif index == 44:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['close'] < (dataframe['ema_16'] * self.buy_44_ma_offset))
                    item_buy_logic.append(dataframe['ewo'] < self.buy_44_ewo_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_44_cti_20_max)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_44_crsi_1h_min)

                # Condition #45 - Long mode
                elif index == 45:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['bb40_2_low'].shift().gt(0))
                    item_buy_logic.append(dataframe['bb40_2_delta'].gt(dataframe['close'] * self.buy_45_bb40_bbdelta_close))
                    item_buy_logic.append(dataframe['closedelta'].gt(dataframe['close'] * self.buy_45_bb40_closedelta_close))
                    item_buy_logic.append(dataframe['tail'].lt(dataframe['bb40_2_delta'] * self.buy_45_bb40_tail_bbdelta))
                    item_buy_logic.append(dataframe['close'].lt(dataframe['bb40_2_low'].shift()))
                    item_buy_logic.append(dataframe['close'].le(dataframe['close'].shift()))
                    item_buy_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_45_ma_offset)
                    item_buy_logic.append(dataframe['ewo'] > self.buy_45_ewo_min)
                    item_buy_logic.append(dataframe['ewo_1h'] > self.buy_45_ewo_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_45_cti_20_1h_max)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_45_r_480_1h_max)                    

            # Condition #46 - Long mode
                elif index == 46:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
                    item_buy_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_46_ema_open_mult))
                    item_buy_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
                    item_buy_logic.append(dataframe['ewo_1h'] > self.buy_46_ewo_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_46_cti_20_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_46_cti_20_1h_max)

                # Condition #47 - Long mode
                elif index == 47:
                    # Non-Standard protections

                    # Logic
                    item_buy_logic.append(dataframe['ewo'] > self.buy_47_ewo_min)
                    item_buy_logic.append(dataframe['close'] < (dataframe['sma_30'] * self.buy_47_ma_offset))
                    item_buy_logic.append(dataframe['rsi_14'] < self.buy_47_rsi_14_max)
                    item_buy_logic.append(dataframe['cti_20'] < self.buy_47_cti_20_max)
                    item_buy_logic.append(dataframe['r_14'] < self.buy_47_r_14_max)
                    item_buy_logic.append(dataframe['ewo_1h'] > self.buy_47_ewo_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_47_cti_20_1h_min)
                    item_buy_logic.append(dataframe['cti_20_1h'] < self.buy_47_cti_20_1h_max)

                # Condition #48 - Uptrend mode
                elif index == 48:
                    # Non-Standard protections
                    item_buy_logic.append(dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12))
                    item_buy_logic.append(dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24))
                    item_buy_logic.append(dataframe['ema_vwma_osc_32'] > 0)
                    item_buy_logic.append(dataframe['ema_vwma_osc_64'] > 0)
                    item_buy_logic.append(dataframe['ema_vwma_osc_96'] > 0)

                    # Logic
                    item_buy_logic.append(dataframe['ewo'] > self.buy_48_ewo_min)
                    item_buy_logic.append(dataframe['ewo_1h'] > self.buy_48_ewo_1h_min)
                    item_buy_logic.append(dataframe['r_480'] > self.buy_48_r_480_min)
                    item_buy_logic.append(dataframe['r_480_1h'] > self.buy_48_r_480_1h_min)
                    item_buy_logic.append(dataframe['r_480_1h'] < self.buy_48_r_480_1h_max)
                    item_buy_logic.append(dataframe['r_480_1h'] > dataframe['r_480_1h'].shift(12))
                    item_buy_logic.append(dataframe['cti_20_1h'] > self.buy_48_cti_20_1h_min)
                    item_buy_logic.append(dataframe['crsi_1h'] > self.buy_48_crsi_1h_min)
                    item_buy_logic.append(dataframe['cti_20'].shift(1).rolling(12).min() < -0.5)
                    item_buy_logic.append(dataframe['cti_20'].shift(1).rolling(12).max() < 0.0)
                    item_buy_logic.append(dataframe['cti_20'].shift(1) < 0.0)
                    item_buy_logic.append(dataframe['cti_20'] > 0.0)

                item_buy = reduce(lambda x, y: x & y, item_buy_logic)
                dataframe.loc[item_buy, 'buy_tag'] += f"{index} "
                conditions.append(item_buy)

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

    def is_support(self, row_data) -> bool:
        return row_data[0] > row_data[1] and row_data[1] > row_data[2] and row_data[2] < row_data[3] and row_data[3] < row_data[4]

    def is_resistance(self, row_data) -> bool:
        return row_data[0] < row_data[1] and row_data[1] < row_data[2] and row_data[2] > row_data[3] and row_data[3] > row_data[4]

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
