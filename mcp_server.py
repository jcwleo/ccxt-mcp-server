# mcp_server.py

import json
from typing import Optional, List, Union, Dict, Annotated, Literal
import ccxt.async_support as ccxtasync # Changed for asynchronous support and alias
from fastmcp import FastMCP
import asyncio
from pydantic import Field
import pandas as pd # Added for DataFrame manipulation

# --- Constants for CCXT Exception Handling ---
CCXT_GENERAL_EXCEPTIONS = (
    ccxtasync.AuthenticationError, # Covers PermissionDenied, AccountNotEnabled, AccountSuspended
    ccxtasync.ArgumentsRequired,
    ccxtasync.BadRequest,          # Covers BadSymbol
    ccxtasync.InsufficientFunds,
    ccxtasync.InvalidAddress,      # Covers AddressPending
    ccxtasync.InvalidOrder,        # Covers OrderNotFound, OrderNotCached, etc.
    # NotSupported is handled specif
    ccxtasync.NetworkError,        # Covers DDoSProtection, RateLimitExceeded, ExchangeNotAvailable, InvalidNonce, RequestTimeout, OnMaintenance, ChecksumError
    ccxtasync.BadResponse,         # Covers NullResponse
    ccxtasync.CancelPending,
    ccxtasync.ExchangeError,       # General ccxt exchange error, placed after more specific ones
    ValueError
)
TimeframeLiteral = Literal[
    '1m', '3m', '5m', '15m', '30m', 
    '1h', '2h', '4h', '6h', '8h', '12h', 
    '1d', '3d', '1w', '1M'
]

# Initialize FastMCP
mcp = FastMCP("CCXT MCP Server 🚀")

import pandas as pd
from typing import Dict, Optional, Tuple
import numpy as np

def compute_rsi(df: pd.DataFrame, length: int = 14, price_source: str = 'close') -> Optional[pd.Series]:
    """Calculates Relative Strength Index (RSI) using pandas.
    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        length: The period for RSI calculation.
        price_source: The DataFrame column to use for price (e.g., 'close', 'hlc3').
    Returns:
        Pandas Series with RSI values, or None if calculation fails.
    """
    if price_source not in df.columns:
        raise ValueError(f"Price source column '{price_source}' not found in DataFrame.")
    if df[price_source].isnull().all():
        # print(f"Warning: Price source column '{price_source}' for RSI is all NaN.")
        return None
    try:
        delta = df[price_source].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate initial average gain and loss using SMA for the first period
        avg_gain = gain.rolling(window=length, min_periods=length).mean()
        avg_loss = loss.rolling(window=length, min_periods=length).mean()

        # For subsequent periods, use Wilder's smoothing method (equivalent to EMA with alpha = 1/length)
        # For pandas EWM, alpha = 2 / (span + 1), so span = (2 / alpha) - 1 = 2*length - 1
        # However, it's more direct to use the recursive formula after the first value.
        
        # Fill NaN for the first `length` periods because rolling mean needs `length` values
        # For the very first RSI value, avg_gain and avg_loss are simple averages.
        # Subsequent values are smoothed.

        for i in range(length, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (length - 1) + gain.iloc[i]) / length
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (length - 1) + loss.iloc[i]) / length

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # RSI can be NaN if avg_loss is 0.
        # If avg_loss is 0 and avg_gain is also 0, rs is NaN, rsi is NaN.
        # If avg_loss is 0 and avg_gain is > 0, rs is inf, 100 / (1 + inf) is 0, so RSI is 100.
        # To maintain consistency with other indicators that have initial NaNs,
        # we will let NaNs propagate and handle them during the first_valid_index logic.
        # rsi.fillna(100, inplace=True) # Removed: Let initial NaNs remain
        # rsi[avg_loss == 0] = 100 # Removed for consistency, will be NaN if rs is NaN or inf if avg_loss is 0.
                                  # Or, if we want to be strict to definition:
        rsi.loc[avg_loss == 0] = 100.0 # Set to 100 where avg_loss is 0 and avg_gain > 0 (rs is inf)
        # If both are 0, rs is nan, rsi remains nan. This is fine.
        
        return rsi

    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None

def compute_sma(df: pd.DataFrame, length: int = 20, price_source: str = 'close') -> Optional[pd.Series]:
    """Calculates Simple Moving Average (SMA) using pandas.
    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        length: The period for SMA calculation.
        price_source: The DataFrame column to use for price (e.g., 'close', 'hlc3').
    Returns:
        Pandas Series with SMA values, or None if calculation fails.
    """
    if price_source not in df.columns:
        raise ValueError(f"Price source column '{price_source}' not found in DataFrame.")
    if df[price_source].isnull().all():
        # print(f"Warning: Price source column '{price_source}' for SMA is all NaN.")
        return None
    try:
        sma_series = df[price_source].rolling(window=length, min_periods=length).mean()
        return sma_series
    except Exception as e:
        print(f"Error calculating SMA: {e}")
        return None

def compute_ema(df: pd.DataFrame, length: int = 20, price_source: str = 'close') -> Optional[pd.Series]:
    """Calculates Exponential Moving Average (EMA) using pandas.
    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        length: The span for EMA calculation.
        price_source: The DataFrame column to use for price (e.g., 'close', 'hlc3').
    Returns:
        Pandas Series with EMA values, or None if calculation fails.
    """
    if price_source not in df.columns:
        raise ValueError(f"Price source column '{price_source}' not found in DataFrame.")
    if df[price_source].isnull().all():
        # print(f"Warning: Price source column '{price_source}' for EMA is all NaN.")
        return None
    try:
        ema_series = df[price_source].ewm(span=length, adjust=False, min_periods=length).mean()
        return ema_series
    except Exception as e:
        print(f"Error calculating EMA: {e}")
        return None

def compute_macd(
    df: pd.DataFrame, 
    fast_length: int = 12, 
    slow_length: int = 26, 
    signal_length: int = 9, 
    price_source: str = 'close'
) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
    """Calculates Moving Average Convergence Divergence (MACD) using pandas.
    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        fast_length: The period for the fast EMA.
        slow_length: The period for the slow EMA.
        signal_length: The period for the signal line EMA.
        price_source: The DataFrame column to use for price.
    Returns:
        A tuple of (macd_line, signal_line, histogram), or None if calculation fails.
    """
    if price_source not in df.columns:
        raise ValueError(f"Price source column '{price_source}' not found in DataFrame.")

    try:
        ema_fast = df[price_source].ewm(span=fast_length, adjust=False, min_periods=fast_length).mean()
        ema_slow = df[price_source].ewm(span=slow_length, adjust=False, min_periods=slow_length).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_length, adjust=False, min_periods=signal_length).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return None

def compute_bbands(
    df: pd.DataFrame, 
    length: int = 20, 
    std_dev: float = 2.0, 
    price_source: str = 'close'
) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
    """Calculates Bollinger Bands (BBANDS) using pandas.
    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        length: The period for the middle band (SMA) and standard deviation.
        std_dev: The number of standard deviations for the upper and lower bands.
        price_source: The DataFrame column to use for price.
    Returns:
        A tuple of (lower_band, middle_band, upper_band), or None if calculation fails.
    """
    if price_source not in df.columns:
        raise ValueError(f"Price source column '{price_source}' not found in DataFrame.")

    try:
        middle_band = df[price_source].rolling(window=length, min_periods=length).mean()
        rolling_std = df[price_source].rolling(window=length, min_periods=length).std()
        
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return lower_band, middle_band, upper_band
    except Exception as e:
        print(f"Error calculating BBANDS: {e}")
        return None

def compute_stochastic_oscillator(
    df: pd.DataFrame, 
    k_period: int = 14, 
    d_period: int = 3, 
    smooth_k: int = 3, 
    price_source_high: str = 'high', 
    price_source_low: str = 'low', 
    price_source_close: str = 'close'
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """
    Calculates the Stochastic Oscillator (%K and %D).

    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        k_period: The look-back period for the K calculation.
        d_period: The period for the D line (SMA of %K).
        smooth_k: The smoothing period for %K (SMA of raw %K).
        price_source_high: DataFrame column for high prices.
        price_source_low: DataFrame column for low prices.
        price_source_close: DataFrame column for close prices.

    Returns:
        A tuple of (percent_k, percent_d) pandas Series, or None if calculation fails.
    """
    if df.empty:
        # print("Warning: DataFrame is empty for Stochastic Oscillator calculation.")
        return None

    required_cols = [price_source_high, price_source_low, price_source_close]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Price source column '{col}' not found in DataFrame.")
        if df[col].isnull().all():
            # print(f"Warning: Price source column '{col}' for Stochastic Oscillator is all NaN.")
            return None
            
    try:
        lowest_low = df[price_source_low].rolling(window=k_period, min_periods=k_period).min()
        highest_high = df[price_source_high].rolling(window=k_period, min_periods=k_period).max()

        delta_high_low = highest_high - lowest_low
        
        # Calculate raw %K
        # Set to 50 if delta_high_low is 0 (flat price in k_period)
        # otherwise calculate 100 * ((close - lowest_low) / delta_high_low)
        raw_k_values = np.where(
            delta_high_low == 0, 
            50.0,  # Set to 50 if no range (highest_high == lowest_low)
            100 * ((df[price_source_close] - lowest_low) / delta_high_low)
        )
        raw_k = pd.Series(raw_k_values, index=df.index)
        
        # Handle cases where raw_k might still be NaN due to NaNs in input even if delta_high_low is not 0
        # For example, if close, lowest_low, or highest_high had NaNs not caught by min_periods.
        # Or if (close - lowest_low) is NaN / non-zero_delta is NaN.
        # A common practice is to fill these with a mid-value or propagate.
        # Given the np.where, the main source of NaNs would be if inputs to np.where are NaN.
        # Rolling functions with min_periods handle initial NaNs.
        # If raw_k has NaNs after np.where, it means some input to the calculation was NaN.
        # We can fill these with 50, or propagate. Propagating is often safer.
        # However, the problem description implies filling NaNs from 0/0 with 50.
        # The `np.where(delta_high_low == 0, 50.0, ...)` handles the 0/0 case explicitly.
        # NaNs resulting from other operations (e.g. NaN in close) should ideally propagate.

        if smooth_k > 1:
            percent_k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        else:
            percent_k = raw_k

        percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
        
        # Ensure no leading NaNs beyond what's necessary due to rolling windows
        # This is generally handled by the rolling(min_periods=...)
        # and how process_indicator_series (if used externally) would pick first_valid_index.

        return percent_k, percent_d

    except Exception as e:
        print(f"Error calculating Stochastic Oscillator: {e}")
        return None

def compute_atr(
    df: pd.DataFrame, 
    period: int = 14, 
    price_source_high: str = 'high', 
    price_source_low: str = 'low', 
    price_source_close: str = 'close'
) -> Optional[pd.Series]:
    """
    Calculates the Average True Range (ATR).

    Args:
        df: Pandas DataFrame with OHLCV data, indexed by timestamp.
        period: The look-back period for ATR calculation.
        price_source_high: DataFrame column for high prices.
        price_source_low: DataFrame column for low prices.
        price_source_close: DataFrame column for close prices.

    Returns:
        A pandas Series with ATR values, or None if calculation fails.
    """
    if df.empty or len(df) < 1: # Check for empty or too short DataFrame
        # print("Warning: DataFrame is empty or too short for ATR calculation.")
        return None

    required_cols = [price_source_high, price_source_low, price_source_close]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Price source column '{col}' not found in DataFrame.")
        if df[col].isnull().all():
            # print(f"Warning: Price source column '{col}' for ATR is all NaN.")
            return None
            
    # Check if there's enough data for at least one ATR value after considering the period
    # While ewm with min_periods handles this by returning NaNs, 
    # an explicit check for len(df) < period could be done here if strictness is desired.
    # However, typically, if len(df) is 1, TR can be H-L, but ATR would be NaN until `period` TR values exist.
    # The current logic with min_periods=period in ewm is standard.
    
    try:
        high_col = df[price_source_high]
        low_col = df[price_source_low]
        close_col = df[price_source_close]

        high_low = high_col - low_col
        high_prev_close = (high_col - close_col.shift(1)).abs()
        low_prev_close = (low_col - close_col.shift(1)).abs()

        # Create a DataFrame for TR components
        # Ensure index alignment, especially if inputs had different NaNs initially
        tr_components = [high_low, high_prev_close, low_prev_close]
        # Filter out series that are all NaN before concat, to avoid issues if a price source was valid but led to all NaN here
        tr_components_filtered = [s for s in tr_components if not s.isnull().all()]
        
        if not tr_components_filtered: # Should not happen if input column checks passed
            return None

        tr_df = pd.concat(tr_components_filtered, axis=1)
        true_range = tr_df.max(axis=1, skipna=False) # skipna=False to ensure NaNs propagate if all components are NaN for a row

        # Handle the first TR value: TR1 = High1 - Low1
        # .iat requires integer index, ensure df is not empty (already checked)
        if len(df) > 0: # Redundant due to earlier check, but safe
             true_range.iat[0] = high_col.iat[0] - low_col.iat[0]
        
        # Calculate ATR using Wilder's Smoothing (approximated by EWM with adjust=False)
        # min_periods=period ensures that ATR is NaN until there are `period` TR values.
        # The first ATR value will be the SMA of the first `period` TR values.
        # Subsequent values use the EMA formula.
        # Pandas ewm with adjust=False and alpha = 1/N directly implements Wilder's smoothing.
        atr = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        
        return atr

    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return None

# --- Helper Function to Initialize CCXT Exchange ---
async def get_exchange_instance(
    exchange_id: str, 
    api_key_info: Optional[Dict[str, str]] = None,
    exchange_config_options: Optional[Dict] = None # Added to handle options like defaultType
) -> ccxtasync.Exchange:
    """
    Asynchronously initializes and returns a CCXT exchange instance.
    This function serves as a common utility to create authenticated or unauthenticated
    exchange instances for interacting with various cryptocurrency exchanges.

    (Note: Caching of instances is not currently implemented in this helper).

    If `api_key_info` is provided, an authenticated instance is created,
    suitable for private API calls (e.g., trading, balance fetching).
    Otherwise, an unauthenticated instance is returned, suitable for public API calls
    (e.g., fetching market data, tickers).

    Args:
        exchange_id: The lowercase string ID of the exchange (e.g., 'binance', 'kucoin', 'upbit').
                     This ID is used to dynamically load the appropriate CCXT exchange class.
        api_key_info: Optional dictionary containing API credentials.
                      Expected keys: 'apiKey', 'secret'.
                      Some exchanges might also require a 'password' (for passphrase).
                      Example: `{'apiKey': 'YOUR_API_KEY', 'secret': 'YOUR_SECRET'}`
        exchange_config_options: Optional dictionary for CCXT client configurations.
                                 This is crucial for specifying market types (e.g., spot, futures, options)
                                 or other exchange-specific settings.
                                 Example: `{'defaultType': 'future'}` for futures trading,
                                          `{'options': {'adjustForTimeDifference': True}}` for time sync.

    Returns:
        An initialized asynchronous CCXT exchange instance (`ccxtasync.Exchange`).

    Raises:
        ccxtasync.ExchangeNotFound: If the `exchange_id` does not correspond to a supported
                                     exchange in the `ccxtasync` library.
    """
    exchange_id_lower = exchange_id.lower()
    try:
        exchange_class = getattr(ccxtasync, exchange_id_lower)
    except AttributeError:
        raise ccxtasync.ExchangeNotFound(f"Exchange '{exchange_id_lower}' not found in ccxtasync library.")
    
    config = {
        'enableRateLimit': True,
        # 'verbose': True, # 디버깅 시 유용
    }
    if api_key_info:
        config.update(api_key_info)
    
    if exchange_config_options: # Merge additional exchange-specific config options
        config.update(exchange_config_options)
        
    instance = exchange_class(config)
    return instance

# --- MCP Tools for CCXT Functions (Async) ---

# Note: All tools now accept optional api_key, secret_key, and passphrase.
# However, tools performing private actions (e.g., fetching balance, creating orders)
# will return an error internally if these are not provided.

@mcp.tool(
    name="fetch_account_balance",
    description="Fetches the current balance of an account from a specified cryptocurrency exchange. "
                "API authentication (api_key, secret_key) is handled externally. "
                "Use the `params` argument to specify account type (e.g., spot, margin, futures) if the exchange requires it, "
                "or to pass other exchange-specific parameters for fetching balances.",
    tags={"account", "balance", "wallet", "funds", "private", "spot", "margin", "futures", "swap", "unified"}
)
async def fetch_balance_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'coinbasepro', 'upbit'). Case-insensitive.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange (e.g., for KuCoin, OKX). Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `fetchBalance` call or for CCXT client instantiation. "
                                                        "Use this to specify market types (e.g., `{'type': 'margin'}` or `{'options': {'defaultType': 'future'}}`), "
                                                        "or pass other exchange-specific arguments. "
                                                        "Example: `{'type': 'funding'}` or `{'options': {'defaultType': 'swap'}, 'symbol': 'BTC/USDT:USDT'}` for specific balance types.")] = None
) -> Dict:
    """Internal use: Fetches account balance. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for fetch_account_balance."}
    
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase
    
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchBalance']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchBalance."}
        balance = await exchange.fetchBalance(params=tool_params)
        return balance
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e: # Example of specific handling if needed, though covered by general if not separately handled
        return {"error": f"Operation Not Supported: {str(e)}"} # More specific message
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_account_balance: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_deposit_address",
    description="Fetches the deposit address for a specific cryptocurrency on a given exchange. "
                "API authentication (api_key, secret_key) is handled externally. "
                "The `params` argument can be used to specify the network or chain if the currency supports multiple (e.g., ERC20, TRC20).",
    tags={"account", "deposit", "address", "funding", "receive", "private", "crypto"}
)
async def fetch_deposit_address_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'kraken'). Case-insensitive.")],
    code: Annotated[str, Field(description="Currency code to fetch the deposit address for (e.g., 'BTC', 'ETH', 'USDT').")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `fetchDepositAddress` call or for client instantiation. "
                                                        "Crucially, use this to specify the network/chain if the cryptocurrency exists on multiple networks. "
                                                        "Example: `{'network': 'TRC20'}` for USDT on Tron network, or `{'chain': 'BEP20'}`. "
                                                        "Can also include `{'options': ...}` for client-specific settings if needed.")] = None
) -> Dict:
    """Internal use: Fetches deposit address. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for fetch_deposit_address."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase
        
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchDepositAddress']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchDepositAddress."}
        address_info = await exchange.fetchDepositAddress(code, params=tool_params)
        return address_info
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_deposit_address: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="withdraw_cryptocurrency",
    description="Initiates a cryptocurrency withdrawal to a specified address. "
                "API authentication (api_key, secret_key) and withdrawal permissions on the API key are handled externally. "
                "Use `params` to specify the network/chain if required by the exchange or currency, and for any other exchange-specific withdrawal parameters.",
    tags={"account", "withdrawal", "transaction", "send", "crypto", "private"}
)
async def withdraw_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'ftx'). Case-insensitive.")],
    code: Annotated[str, Field(description="Currency code for the withdrawal (e.g., 'BTC', 'ETH', 'USDT').")],
    amount: Annotated[float, Field(description="The amount of currency to withdraw. Must be greater than 0.", gt=0)],
    address: Annotated[str, Field(description="The destination address for the withdrawal.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key with withdrawal permissions. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the API. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    tag: Annotated[Optional[str], Field(description="Optional: Destination tag, memo, or payment ID for certain currencies (e.g., XRP, XLM, EOS). Check exchange/currency requirements.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for withdrawals. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `withdraw` call or for client instantiation. "
                                                        "Use this to specify the network/chain (e.g., `{'network': 'BEP20'}`), especially if the currency supports multiple. "
                                                        "May also be used for two-factor authentication codes if supported/required by the exchange via CCXT, or other specific withdrawal options. "
                                                        "Example: `{'network': 'TRC20', 'feeToUser': False}`")] = None
) -> Dict:
    """Internal use: Withdraws cryptocurrency. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for withdraw_cryptocurrency."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['withdraw']:
            return {"error": f"Exchange '{exchange_id}' does not support withdraw."}
        
        withdrawal_info = await exchange.withdraw(code, amount, address, tag, params=tool_params)
        return withdrawal_info
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in withdraw_cryptocurrency: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_open_positions",
    description="Fetches currently open positions for futures, swaps, or other derivatives from an exchange. "
                "API authentication (api_key, secret_key) is handled externally. "
                "CRITICAL: The CCXT client MUST be initialized for the correct market type (e.g., futures, swap) using `params`. "
                "For example, pass `{'options': {'defaultType': 'future'}}` or `{'options': {'defaultType': 'swap'}}` in `params` if not default for the exchange.",
    tags={"account", "positions", "futures", "derivatives", "swap", "margin_trading", "private"}
)
async def fetch_positions_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange that supports derivatives trading (e.g., 'binance', 'bybit', 'okx'). Case-insensitive.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `fetchPositions` call AND for CCXT client instantiation. "
                                                        "CRITICAL for client setup: Include `{'options': {'defaultType': 'future'}}` (or 'swap', 'linear', 'inverse') to specify market type if not the exchange default. "
                                                        "For the API call: Can be used to filter positions by symbol(s) if supported by the exchange (e.g., `{'symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT']}`). "
                                                        "Example for client init: `{'options': {'defaultType': 'future'}}`. Example for call: `{'symbol': 'BTC/USDT:USDT'}`")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches open positions. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for fetch_open_positions."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has.get('fetchPositions'):
            return {"error": f"Exchange '{exchange_id}' may not support fetchPositions or requires specific market type configuration (e.g., {{'options': {{'defaultType': 'future'}}}} passed at client instantiation)."}

        positions = await exchange.fetchPositions(params=tool_params)
        return positions
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e: # Catch NotSupported here if it wasn't in CCXT_GENERAL_EXCEPTIONS
        return {"error": f"FetchPositions Not Supported or requires specific config: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_open_positions: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="set_trading_leverage",
    description="Sets the leverage for a specific trading symbol, typically in futures or margin markets. "
                "API authentication (api_key, secret_key) is handled externally. "
                "CRITICAL: Ensure the CCXT client is initialized for the correct market type (e.g., futures, margin) using `params` (e.g., `{'options': {'defaultType': 'future'}}`). "
                "The `symbol` parameter may or may not be required depending on the exchange and whether setting leverage for all symbols or a specific one.",
    tags={"trading", "leverage", "futures", "margin", "derivatives", "private"}
)
async def set_leverage_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'ftx'). Case-insensitive.")],
    leverage: Annotated[int, Field(description="The desired leverage multiplier (e.g., 10 for 10x). Must be greater than 0.", gt=0)],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional/Required: The symbol (e.g., 'BTC/USDT:USDT' for futures, 'BTC/USDT' for margin) to set leverage for. "
                                                       "Some exchanges require it, others set it account-wide or per market type. Check exchange documentation.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `setLeverage` call AND for CCXT client instantiation. "
                                                        "CRITICAL for client setup: Include `{'options': {'defaultType': 'future'}}` or `{'options': {'defaultType': 'margin'}}` if applicable. "
                                                        "For the API call: May include parameters like `{'marginMode': 'isolated'}` or `{'marginMode': 'cross'}` if supported. "
                                                        "Example for client init: `{'options': {'defaultType': 'future'}}`. Example for call: `{'marginMode': 'isolated'}`")] = None
) -> Dict:
    """Internal use: Sets trading leverage. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for set_trading_leverage."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase
    
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        response = await exchange.setLeverage(leverage, symbol, params=tool_params)
        return response
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e: # Keep NotSupported specific for this as per original logic
        return {"error": f"Exchange '{exchange_id}' does not support the unified setLeverage with the provided arguments. Error: {str(e)}. You may need to use exchange-specific 'params' or check symbol requirements."}
    except CCXT_GENERAL_EXCEPTIONS as e: # Catch other general exceptions after specific NotSupported
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred in set_trading_leverage: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

# --- Tools for Public/Unauthenticated CCXT Functions ---

@mcp.tool(
    name="fetch_ohlcv",
    description="Fetches historical Open-High-Low-Close-Volume (OHLCV) candlestick data for a specific trading symbol and timeframe. "
                "Authentication (api_key, secret_key) is optional; some exchanges might provide more data or higher rate limits with authentication. "
                "Use `params` for exchange-specific options, like requesting 'mark' or 'index' price OHLCV for derivatives, or to set `defaultType` for client instantiation if fetching for non-spot markets.",
    tags={"market_data", "ohlcv", "candles", "candlestick", "chart", "historical_data", "public", "private", "spot", "futures", "swap"}
)
async def fetch_ohlcv_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'kraken'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The trading symbol to fetch OHLCV data for (e.g., 'BTC/USDT', 'ETH/BTC', 'BTC/USDT:USDT' for futures).")],
    timeframe: Annotated[str, Field(description="The length of time each candle represents (e.g., '1m', '5m', '1h', '1d', '1w'). Check exchange for supported timeframes.")],
    since: Annotated[Optional[int], Field(description="Optional: The earliest time in milliseconds (UTC epoch) to fetch OHLCV data from (e.g., 1502962800000 for 2017-08-17T10:00:00Z).", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: The maximum number of OHLCV candles to return. Check exchange for default and maximum limits.", gt=0)] = None,
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `fetchOHLCV` call or for client instantiation. "
                                                        "For client init (if fetching non-spot): `{'options': {'defaultType': 'future'}}`. "
                                                        "For API call: To specify price type for derivatives (e.g., `{'price': 'mark'}` or `{'price': 'index'}`) or other exchange-specific query params. "
                                                        "Example for mark price candles: `{'options': {'defaultType': 'future'}, 'price': 'mark'}`")] = None
) -> Union[List[List[Union[int, float]]], Dict]:
    """Internal use: Fetches OHLCV data. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchOHLCV']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchOHLCV."}
        
        ohlcv_data = await exchange.fetchOHLCV(symbol, timeframe, since, limit, params=tool_params)
        return ohlcv_data
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_ohlcv: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_funding_rate",
    description="Fetches the current or historical funding rate for a perpetual futures contract symbol. "
                "Authentication is optional. "
                "CRITICAL: For many exchanges, the CCXT client must be initialized for futures/swap markets using `params` (e.g., `{'options': {'defaultType': 'future'}}`). "
                "If `fetchFundingRate` is not supported, the exchange might support `fetchFundingRates` (plural) for multiple symbols or historical rates; check error messages or use a more specific tool if available.",
    tags={"market_data", "funding_rate", "futures", "swap", "perpetual", "public", "private"}
)
async def fetch_funding_rate_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'bybit'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The symbol to fetch the funding rate for (e.g., 'BTC/USDT:USDT', 'ETH-PERP'). Ensure correct perpetual contract symbol format for the exchange.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `fetchFundingRate` call or client instantiation. "
                                                        "CRITICAL for client setup: Include `{'options': {'defaultType': 'future'}}` or `{'options': {'defaultType': 'swap'}}` for correct market type. "
                                                        "For API call: May be used for historical rates if supported (e.g., `{'since': timestamp, 'limit': N}`). "
                                                        "Example for client init: `{'options': {'defaultType': 'future'}}`")] = None
) -> Dict:
    """Internal use: Fetches funding rate. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchFundingRate']:
            if exchange.has['fetchFundingRates']:
                 return {"error": f"Exchange '{exchange_id}' supports fetchFundingRates (plural). Try that or check symbol format if fetchFundingRate (singular) is not supported."}
            return {"error": f"Exchange '{exchange_id}' does not support fetchFundingRate."}
        
        funding_rate = await exchange.fetchFundingRate(symbol, params=tool_params)
        return funding_rate
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_funding_rate: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_long_short_ratio",
    description="Fetches the long/short ratio for a symbol, typically for futures markets, by calling exchange-specific (implicit) CCXT methods. "
                "Authentication is optional. Requires specifying the `method_name` and `method_params` within the `params` argument. "
                "Client may need to be initialized for futures/swap markets via `params` (e.g., `{'options': {'defaultType': 'future'}}`).",
    tags={"market_data", "sentiment", "long_short_ratio", "futures", "derivatives", "public", "private", "exchange_specific"}
)
async def fetch_long_short_ratio_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'bybit'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The symbol to fetch the long/short ratio for (e.g., 'BTC/USDT', 'BTC/USDT:USDT'). Format depends on the specific exchange method.")],
    timeframe: Annotated[str, Field(description="Timeframe for the ratio data (e.g., '5m', '1h', '4h', '1d'). Format depends on the specific exchange method.")],
    limit: Annotated[Optional[int], Field(description="Optional: Number of data points to retrieve. Depends on the specific exchange method.", gt=0)] = None,
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="CRUCIAL: Must contain `method_name` (string: the exact CCXT implicit method name, e.g., 'publicGetFuturesDataOpenInterestHist') and `method_params` (dict: arguments for that method). "
                                                        "Can also include `{'options': {'defaultType': 'future'}}` for client instantiation if needed. "
                                                        "Example: `{'options': {'defaultType': 'future'}, 'method_name': 'fapiPublicGetGlobalLongShortAccountRatio', 'method_params': {'period': '5m'}}`")] = None
) -> Dict:
    """Internal use: Fetches long/short ratio. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        
        method_name = tool_params.pop('method_name', None)
        method_args_from_params = tool_params.pop('method_params', {})

        if method_name and hasattr(exchange, method_name):
            call_args = {'symbol': symbol, 'timeframe': timeframe}
            if limit is not None:
                call_args['limit'] = limit
            call_args.update(method_args_from_params)
            
            target_method = getattr(exchange, method_name)
            if asyncio.iscoroutinefunction(target_method):
                 data = await target_method(call_args)
            else: 
                 data = target_method(call_args)
            return data
        else:
            return {"error": f"fetchLongShortRatio is not standard or method_name missing. Exchange '{exchange_id}' may not have '{method_name}'. Provide 'method_name' and 'method_params' in 'params'."}
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e: # If method_name was valid but underlying CCXT call is not supported for the exchange
        return {"error": f"Implicit method call Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_long_short_ratio: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_option_contract_data",
    description="Fetches market data (typically ticker data) for a specific options contract. "
                "Authentication is optional. "
                "For many exchanges, the CCXT client may need to be initialized for options markets using `params` (e.g., `{'options': {'defaultType': 'option'}}`).",
    tags={"market_data", "options", "ticker", "contract_data", "derivatives", "public", "private"}
)
async def fetch_option_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange that supports options trading (e.g., 'deribit', 'okx'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The specific option contract symbol (e.g., 'BTC-28JUN24-70000-C' on Deribit). Format is exchange-specific.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `fetchTicker` (or other relevant fetch calls for options) AND for client instantiation. "
                                                        "For client setup: Include `{'options': {'defaultType': 'option'}}` or similar for correct market type if needed. "
                                                        "For API call: May include exchange-specific params if `fetchTicker` is used or for other option data methods. "
                                                        "Example for client init: `{'options': {'defaultType': 'option'}}`")] = None
) -> Dict:
    """Internal use: Fetches option contract data. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)

        if exchange.has['fetchTicker']:
            option_data = await exchange.fetchTicker(symbol, params=tool_params)
            return option_data
        else:
            return {"error": f"Exchange '{exchange_id}' does not have a standard fetchOption or fetchTicker."}
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_option_contract_data: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_market_ticker",
    description="Fetches the latest ticker data for a specific trading symbol (e.g., price, volume, spread). "
                "Authentication is optional; some exchanges might provide more data or higher rate limits with authentication. "
                "If fetching for non-spot markets (futures, options, swaps), ensure the CCXT client is initialized correctly using `params` (e.g., `{'options': {'defaultType': 'future'}}`).",
    tags={"market_data", "ticker", "price", "last_price", "volume", "public", "private", "spot", "futures", "options", "swap"}
)
async def fetch_ticker_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'coinbase'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The symbol to fetch the ticker for (e.g., 'BTC/USDT', 'ETH/USD', 'BTC/USDT:USDT' for futures, 'BTC-28JUN24-70000-C' for options). Format depends on the market type and exchange.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `fetchTicker` call or for client instantiation. "
                                                        "For client init (if non-spot): `{'options': {'defaultType': 'future'}}` or `{'options': {'defaultType': 'option'}}`. "
                                                        "For API call: May include exchange-specific params if the exchange offers variations on ticker data. "
                                                        "Example for futures ticker: `{'options': {'defaultType': 'future'}}`")] = None
) -> Dict:
    """Internal use: Fetches market ticker. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchTicker']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchTicker."}
        ticker = await exchange.fetchTicker(symbol, params=tool_params)
        return ticker
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_market_ticker: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_public_market_trades",
    description="Fetches recent public trades for a specific trading symbol. Does not require authentication, but providing API keys might increase rate limits or access. "
                "If fetching for non-spot markets (futures, options, swaps), ensure the CCXT client is initialized correctly using `params` (e.g., `{'options': {'defaultType': 'future'}}`).",
    tags={"market_data", "trades", "executions", "history", "public", "private", "spot", "futures", "options", "swap"}
)
async def fetch_trades_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'kraken'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The symbol to fetch public trades for (e.g., 'BTC/USDT', 'ETH/USD', 'BTC/USDT:USDT' for futures). Format depends on the market type and exchange.")],
    since: Annotated[Optional[int], Field(description="Optional: Timestamp in milliseconds (UTC epoch) to fetch trades since (e.g., 1609459200000 for 2021-01-01T00:00:00Z).", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of trades to fetch. Check exchange for default and maximum limits.", gt=0)] = None,
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `fetchTrades` call or for client instantiation. "
                                                        "For client init (if non-spot): `{'options': {'defaultType': 'future'}}` or `{'options': {'defaultType': 'option'}}`. "
                                                        "For API call: May include exchange-specific pagination or filtering parameters. "
                                                        "Example for futures trades: `{'options': {'defaultType': 'future'}}`")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches public market trades. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchTrades']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchTrades."}
        trades = await exchange.fetchTrades(symbol, since, limit, params=tool_params)
        return trades
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Operation Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_public_market_trades: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

# --- Tools Requiring API Authentication ---

@mcp.tool(
    name="create_spot_limit_order",
    description="Places a new limit order in the spot market. "
                "API authentication (api_key, secret_key) and trading permissions on the API key are handled externally. "
                "Use `params` for exchange-specific order parameters like `clientOrderId`, `postOnly`, or time-in-force policies (e.g., `{'timeInForce': 'FOK'}`).",
    tags={"trading", "order", "create", "spot", "limit", "buy", "sell", "private"}
)
async def create_spot_limit_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'coinbasepro'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The spot market symbol to trade (e.g., 'BTC/USDT', 'ETH/BTC').")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' to purchase the base asset, 'sell' to sell it.")],
    amount: Annotated[float, Field(description="The quantity of the base currency to trade. Must be greater than 0.", gt=0)],
    price: Annotated[float, Field(description="The price at which to place the limit order. Must be greater than 0.", gt=0)],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key with trading permissions. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the API. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `createOrder` call. "
                                                        "Common uses include `{'clientOrderId': 'your_custom_id'}` for custom order identification, "
                                                        "or specifying order properties like `{'postOnly': True}` (maker-only) or time-in-force policies (e.g., `{'timeInForce': 'GTC' / 'IOC' / 'FOK'}`). "
                                                        "Example: `{'clientOrderId': 'my_spot_order_123', 'timeInForce': 'FOK'}`. "
                                                        "No `options` for client instantiation are typically needed for spot orders unless the exchange has specific requirements.")] = None
) -> Dict:
    """Internal use: Creates a spot limit order. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for create_spot_limit_order."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['createOrder']:
            # Even if createOrder is marked as false, specific order methods might exist.
            # We'll rely on hasattr for the specific methods.
            pass

        if side == "buy":
            if not hasattr(exchange, 'create_limit_buy_order'):
                return {"error": f"Exchange '{exchange_id}' does not support create_limit_buy_order via a dedicated method. Falling back to createOrder."}
            order = await exchange.create_limit_buy_order(symbol, amount, price, params=tool_params)
        elif side == "sell":
            if not hasattr(exchange, 'create_limit_sell_order'):
                return {"error": f"Exchange '{exchange_id}' does not support create_limit_sell_order via a dedicated method. Falling back to createOrder."}
            order = await exchange.create_limit_sell_order(symbol, amount, price, params=tool_params)
        else:
            return {"error": f"Invalid side: {side}. Must be 'buy' or 'sell'."}
            
        # order = await exchange.createOrder(symbol, 'limit', side, amount, price, params=tool_params)
        return order
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Order creation Not Supported: {str(e)}"}    
    except Exception as e:
        return {"error": f"An unexpected error occurred in create_spot_limit_order: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="create_spot_market_order",
    description="Places a new market order in the spot market, to be filled at the best available current price. "
                "API authentication (api_key, secret_key) and trading permissions on the API key are handled externally. "
                "Use `params` for exchange-specific order parameters like `clientOrderId` or quote order quantity (if supported).",
    tags={"trading", "order", "create", "spot", "market", "buy", "sell", "private"}
)
async def create_spot_market_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'kraken'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The spot market symbol to trade (e.g., 'BTC/USDT', 'ETH/EUR').")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' to purchase the base asset, 'sell' to sell it.")],
    amount: Annotated[float, Field(description="The quantity of the base currency to trade (for a market buy, unless 'createMarketBuyOrderRequiresPrice' is False, then it's the quote currency amount for some exchanges like Upbit) or the quantity to sell (for a market sell). Must be greater than 0.", gt=0)],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key with trading permissions. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the API. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `createOrder` call. "
                                                        "Common uses include `{'clientOrderId': 'your_custom_id'}`. "
                                                        "For market buy orders, some exchanges allow `{'quoteOrderQty': quote_amount}` to specify the amount in quote currency (e.g., spend 100 USDT on BTC). "
                                                        "For exchanges like Upbit market buy, you might need to pass `{'createMarketBuyOrderRequiresPrice': False}` if `amount` represents the total cost in quote currency. "
                                                        "Example: `{'clientOrderId': 'my_market_buy_001', 'quoteOrderQty': 100}`. "
                                                        "No `options` for client instantiation are typically needed for spot orders.")] = None
) -> Dict:
    """Internal use: Creates a spot market order. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for create_spot_market_order."}
        
    tool_params = params.copy() if params else {}
    # For Upbit market buy orders, 'amount' is the total cost in KRW.
    # CCXT requires 'createMarketBuyOrderRequiresPrice': False to be set in params.
    if exchange_id.lower() == 'upbit' and side == 'buy':
        tool_params['createMarketBuyOrderRequiresPrice'] = False

    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['createOrder']:
            # Even if createOrder is marked as false, specific order methods might exist.
            # We'll rely on hasattr for the specific methods.
            pass

        if side == "buy":
            if not hasattr(exchange, 'create_market_buy_order'):
                return {"error": f"Exchange '{exchange_id}' does not support create_market_buy_order via a dedicated method. Falling back to createOrder."}
            order = await exchange.create_market_buy_order(symbol, amount, params=tool_params)
        elif side == "sell":
            if not hasattr(exchange, 'create_market_sell_order'):
                return {"error": f"Exchange '{exchange_id}' does not support create_market_sell_order via a dedicated method. Falling back to createOrder."}
            order = await exchange.create_market_sell_order(symbol, amount, params=tool_params)
        else:
            return {"error": f"Invalid side: {side}. Must be 'buy' or 'sell'."}

        # order = await exchange.createOrder(symbol, 'market', side, amount, params=tool_params)
        return order
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Order creation Not Supported: {str(e)}"}    
    except Exception as e:
        return {"error": f"An unexpected error occurred in create_spot_market_order: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="create_futures_limit_order",
    description="Places a new limit order in a futures/swap market. "
                "API authentication (api_key, secret_key) and trading permissions are handled externally. "
                "CRITICAL: The CCXT client MUST be initialized for the correct market type (e.g., 'future', 'swap') using `params` (e.g., `{'options': {'defaultType': 'future'}}`). "
                "Use `params` also for exchange-specific order parameters like `clientOrderId`, `postOnly`, `reduceOnly`, `timeInForce`.",
    tags={"trading", "order", "create", "futures", "swap", "derivatives", "limit", "buy", "sell", "private"}
)
async def create_futures_limit_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange that supports futures/swap trading (e.g., 'binance', 'bybit', 'okx'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The futures/swap contract symbol to trade (e.g., 'BTC/USDT:USDT', 'ETH-PERP'). Format is exchange-specific.")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' for a long position, 'sell' for a short position.")],
    amount: Annotated[float, Field(description="The quantity of contracts or base currency to trade. Must be greater than 0.", gt=0)],
    price: Annotated[float, Field(description="The price at which to place the limit order. Must be greater than 0.", gt=0)],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key with trading permissions. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the API. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `createOrder` call AND for client instantiation. "
                                                        "CRITICAL for client setup: Include `{'options': {'defaultType': 'future'}}` (or 'swap', 'linear', 'inverse' etc., depending on exchange and contract) to specify market type. "
                                                        "For API call: Common uses include `{'clientOrderId': 'custom_id'}`, `{'postOnly': True}`, `{'reduceOnly': True}`, `{'timeInForce': 'GTC'}`. "
                                                        "Example: `{'options': {'defaultType': 'future'}, 'reduceOnly': True, 'clientOrderId': 'my_fut_limit_001'}`")] = None
) -> Dict:
    """Internal use: Creates a futures limit order. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for create_futures_limit_order."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', {'defaultType': 'future'}) # Default to future if not specified
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['createOrder']:
            return {"error": f"Exchange '{exchange_id}' does not support createOrder for the configured market type."}
        
        order = await exchange.createOrder(symbol, 'limit', side, amount, price, params=tool_params)
        return order
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Order creation Not Supported: {str(e)}"}    
    except Exception as e:
        return {"error": f"An unexpected error occurred in create_futures_limit_order: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="create_futures_market_order",
    description="Places a new market order in a futures/swap market, filled at the best available current price. "
                "API authentication (api_key, secret_key) and trading permissions are handled externally. "
                "CRITICAL: The CCXT client MUST be initialized for the correct market type (e.g., 'future', 'swap') using `params` (e.g., `{'options': {'defaultType': 'future'}}`). "
                "Use `params` also for exchange-specific parameters like `clientOrderId` or `reduceOnly`.",
    tags={"trading", "order", "create", "futures", "swap", "derivatives", "market", "buy", "sell", "private"}
)
async def create_futures_market_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange that supports futures/swap trading (e.g., 'binance', 'bybit'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The futures/swap contract symbol to trade (e.g., 'BTC/USDT:USDT', 'ETH-PERP'). Format is exchange-specific.")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' for a long position, 'sell' for a short position.")],
    amount: Annotated[float, Field(description="The quantity of contracts or base currency to trade. Must be greater than 0.", gt=0)],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key with trading permissions. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the API. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `createOrder` call AND for client instantiation. "
                                                        "CRITICAL for client setup: Include `{'options': {'defaultType': 'future'}}` (or 'swap', etc.) to specify market type. "
                                                        "For API call: Common uses include `{'clientOrderId': 'custom_id'}`, `{'reduceOnly': True}`. "
                                                        "Some exchanges might support `{'quoteOrderQty': quote_amount}` for market buys in quote currency, but this is less common for futures than spot. Check exchange docs. "
                                                        "Example: `{'options': {'defaultType': 'future'}, 'reduceOnly': True, 'clientOrderId': 'my_fut_market_001'}`")] = None
) -> Dict:
    """Internal use: Creates a futures market order. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for create_futures_market_order."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', {'defaultType': 'future'}) # Default to future if not specified
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['createOrder']:
            return {"error": f"Exchange '{exchange_id}' does not support createOrder for the configured market type."}
        
        order = await exchange.createOrder(symbol, 'market', side, amount, params=tool_params)
        return order
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Order creation Not Supported: {str(e)}"}    
    except Exception as e:
        return {"error": f"An unexpected error occurred in create_futures_market_order: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="cancel_order",
    description="Cancels an existing open order on an exchange. "
                "API authentication (api_key, secret_key) is handled externally. "
                "The `symbol` parameter is required by some exchanges, optional for others. "
                "If canceling an order in a non-spot market (futures, options), ensure the CCXT client is initialized correctly using `params` (e.g., `{'options': {'defaultType': 'future'}}`).",
    tags={"trading", "order", "cancel", "manage_order", "private"}
)
async def cancel_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'ftx'). Case-insensitive.")],
    id: Annotated[str, Field(description="The order ID (string) of the order to be canceled.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key with trading permissions. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the API. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional/Required: The symbol of the order (e.g., 'BTC/USDT', 'BTC/USDT:USDT'). "
                                                       "Required by some exchanges for `cancelOrder`, optional for others. Check exchange documentation.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `cancelOrder` call or for client instantiation. "
                                                        "For client init (if non-spot): `{'options': {'defaultType': 'future'}}` or `{'options': {'defaultType': 'option'}}`. "
                                                        "For API call: Some exchanges might accept `clientOrderId` here if the main `id` is the exchange's ID, or other specific flags. "
                                                        "Example for futures order cancel: `{'options': {'defaultType': 'future'}}`")] = None
) -> Dict:
    """Internal use: Cancels an order. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for cancel_order."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['cancelOrder']:
            return {"error": f"Exchange '{exchange_id}' does not support cancelOrder."}
        
        response = await exchange.cancelOrder(id, symbol, params=tool_params)
        return response
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Cancel order Not Supported: {str(e)}"}    
    except Exception as e:
        return {"error": f"An unexpected error occurred in cancel_order: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_order_history",
    description="Fetches a list of your orders (open, closed, canceled, etc.) for an account, optionally filtered by symbol, time, and limit. "
                "API authentication (api_key, secret_key) is handled externally. "
                "If fetching orders from a non-spot market (futures, options), ensure the CCXT client is initialized correctly using `params` (e.g., `{'options': {'defaultType': 'future'}}`). "
                "Some exchanges might use `fetchOrders` to get only open or closed orders by default; use `params` for finer control if supported (e.g. `{'status': 'open'}`).",
    tags={"account", "orders", "history", "trade_history", "manage_order", "private"}
)
async def fetch_orders_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'kucoin'). Case-insensitive.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional: The symbol (e.g., 'BTC/USDT', 'ETH/USDT:USDT') to fetch orders for. If omitted, orders for all symbols may be returned (exchange-dependent).")] = None,
    since: Annotated[Optional[int], Field(description="Optional: Timestamp in milliseconds (UTC epoch) to fetch orders created since this time.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of orders to retrieve. Check exchange for default and maximum limits.", gt=0)] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `fetchOrders` call or for client instantiation. "
                                                        "For client init (if non-spot): `{'options': {'defaultType': 'future'}}`. "
                                                        "For API call: Can be used to filter by order status (e.g., `{'status': 'open'/'closed'/'canceled'}` if supported), order type, or other exchange-specific filters. "
                                                        "Example for open futures orders: `{'options': {'defaultType': 'future'}, 'status': 'open'}`")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches order history. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for fetch_order_history."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchOrders']:
            if exchange.has['fetchOpenOrders'] or exchange.has['fetchClosedOrders']:
                return {"error": f"Exchange '{exchange_id}' does not support fetchOrders directly. Try fetchOpenOrders_tool or fetchClosedOrders_tool if available (not currently implemented as separate tools)."}
            return {"error": f"Exchange '{exchange_id}' does not support fetchOrders."}
        
        orders = await exchange.fetchOrders(symbol, since, limit, params=tool_params)
        return orders
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Fetching orders Not Supported: {str(e)}"}    
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_order_history: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

@mcp.tool(
    name="fetch_my_trade_history",
    description="Fetches the history of your executed trades (fills) for an account, optionally filtered by symbol, time, and limit. "
                "API authentication (api_key, secret_key) is handled externally. "
                "If fetching trades from a non-spot market (futures, options), ensure the CCXT client is initialized correctly using `params` (e.g., `{'options': {'defaultType': 'future'}}`). "
                "Use `params` for any exchange-specific filtering not covered by standard arguments (e.g., filtering by orderId).",
    tags={"account", "trades", "executions", "fills", "history", "trade_history", "private"}
)
async def fetch_my_trades_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'ftx'). Case-insensitive.")],
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key. If not directly provided, the system may use pre-configured credentials. Authentication is required for this operation.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional: The symbol (e.g., 'BTC/USDT', 'BTC/USDT:USDT') to fetch your trades for. If omitted, trades for all symbols may be returned (exchange-dependent).")] = None,
    since: Annotated[Optional[int], Field(description="Optional: Timestamp in milliseconds (UTC epoch) to fetch trades executed since this time.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of trades to retrieve. Check exchange for default and maximum limits.", gt=0)] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT `fetchMyTrades` call or for client instantiation. "
                                                        "For client init (if non-spot): `{'options': {'defaultType': 'future'}}`. "
                                                        "For API call: Can be used for exchange-specific filters like `{'orderId': 'some_order_id'}` to fetch trades for a specific order, or other types of filtering. "
                                                        "Example for trades of a specific futures order: `{'options': {'defaultType': 'future'}, 'orderId': '12345'}`")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches user's trade history. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for fetch_my_trade_history."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['fetchMyTrades']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchMyTrades."}
        
        my_trades = await exchange.fetchMyTrades(symbol, since, limit, params=tool_params)
        return my_trades
    except CCXT_GENERAL_EXCEPTIONS as e:
        return {"error": str(e)}
    except ccxtasync.NotSupported as e:
        return {"error": f"Fetching trades Not Supported: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_my_trade_history: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

# --- 기술적 지표 계산 도구 (리팩토링) ---
# 파라미터 파싱 헬퍼 함수
def parse_indicator_params(indicator_params):
    """지표 파라미터를 파싱하는 함수"""
    parsed_params_dict = {}
    if indicator_params and isinstance(indicator_params, str):
        try:
            parsed_params_dict = json.loads(indicator_params)
        except json.JSONDecodeError:
            # 에러 처리는 상위 함수에서 함
            pass
    elif isinstance(indicator_params, dict):
        parsed_params_dict = indicator_params
    return parsed_params_dict

# OHLCV 데이터 가져오기
async def fetch_ohlcv_data(exchange_id, symbol, timeframe, limit, api_key=None, secret_key=None, passphrase=None, params=None):
    """지정된 거래소에서 OHLCV 데이터를 가져오는 함수"""
    api_key_info_dict = None
    if api_key and secret_key:
        api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
        if passphrase:
            api_key_info_dict['password'] = passphrase
            
    fetch_params = params.copy() if params else {}
    client_config_options = fetch_params.pop('options', None) 

    exchange_instance = None
    try:
        exchange_instance = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange_instance.has['fetchOHLCV']:
            return None, f"Exchange '{exchange_id}' does not support fetchOHLCV."
        
        ohlcv_data = await exchange_instance.fetchOHLCV(symbol, timeframe, limit=limit, params=fetch_params)
        if not ohlcv_data:
            return None, f"No OHLCV data returned for {symbol} on {exchange_id} with timeframe {timeframe}."
        
        return ohlcv_data, None
    except CCXT_GENERAL_EXCEPTIONS as e:
        return None, f"CCXT Error: {str(e)}"
    except ccxtasync.NotSupported as e:
        return None, f"Operation Not Supported: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
    finally:
        if exchange_instance:
            await exchange_instance.close()

# 데이터프레임 준비 함수
def prepare_dataframe(ohlcv_data, price_source_col):
    """OHLCV 데이터를 DataFrame으로 변환하고 가공하는 함수"""
    if not ohlcv_data:
        return None, "Failed to fetch OHLCV data, or no data available for the period."
        
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        return None, "OHLCV data was empty after converting to DataFrame."
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    if df.empty:
        return None, "OHLCV data became empty after cleaning."

    if price_source_col == 'hlc3':
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    elif price_source_col == 'ohlc4':
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    if price_source_col not in df.columns:
        return None, f"Specified 'price_source' column '{price_source_col}' could not be found or derived."
    if df[price_source_col].isnull().all():
        return None, f"The 'price_source' column '{price_source_col}' contains all NaN values."
        
    return df, None

# 시리즈 처리 공통 로직
def process_indicator_series(series, limit):
    """시리즈 데이터를 처리하는 공통 함수"""
    if series is None or series.empty:
        return pd.Series(dtype=float)
        
    first_idx = series.first_valid_index()
    if first_idx is None:
        return pd.Series(dtype=float)
        
    return series.loc[first_idx:].tail(limit)

# 결과 포맷팅 함수들
def timestamp_to_iso(timestamp):
    """Pandas 타임스탬프를 ISO 8601 형식 문자열로 변환"""
    # 밀리초를 제외한 ISO 형식 문자열 변환
    return timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

def format_stochastic_to_list(percent_k, percent_d):
    """Stochastic Oscillator 시리즈를 리스트로 변환하는 함수"""
    if percent_k is None or percent_d is None:
        return []
    
    results = []
    # Ensure both series are aligned and iterate through common indices
    common_index = percent_k.index.intersection(percent_d.index)
    
    for idx in common_index:
        k_val = percent_k.get(idx)
        d_val = percent_d.get(idx)
        results.append({
            "datetime": timestamp_to_iso(idx),
            "percent_k": round(k_val, 4) if pd.notnull(k_val) else None,
            "percent_d": round(d_val, 4) if pd.notnull(d_val) else None,
        })
    return results

def format_series_to_list(series, name):
    """단일 시리즈를 리스트로 변환하는 함수"""
    if series is None or series.empty:
        return []
    return [
        {"datetime": timestamp_to_iso(idx), name: round(val, 4) if pd.notnull(val) else None}
        for idx, val in series.items()
    ]

def format_macd_to_list(macd_line, signal_line, histogram):
    """MACD 시리즈를 리스트로 변환하는 함수"""
    if macd_line is None or signal_line is None or histogram is None:
        return []
    
    results = []
    for i in range(len(macd_line)):
        results.append({
            "datetime": timestamp_to_iso(macd_line.index[i]),
            "macd": round(macd_line.iloc[i], 4) if pd.notnull(macd_line.iloc[i]) else None,
            "signal": round(signal_line.iloc[i], 4) if pd.notnull(signal_line.iloc[i]) else None,
            "histogram": round(histogram.iloc[i], 4) if pd.notnull(histogram.iloc[i]) else None,
        })
    return results

def format_bbands_to_list(lower, middle, upper):
    """볼린저밴드 시리즈를 리스트로 변환하는 함수"""
    if lower is None or middle is None or upper is None:
        return []
    
    results = []
    for i in range(len(middle)):
        results.append({
            "datetime": timestamp_to_iso(middle.index[i]),
            "lower": round(lower.iloc[i], 4) if pd.notnull(lower.iloc[i]) else None,
            "middle": round(middle.iloc[i], 4) if pd.notnull(middle.iloc[i]) else None,
            "upper": round(upper.iloc[i], 4) if pd.notnull(upper.iloc[i]) else None,
        })
    return results

# 지표 계산 함수들
def calculate_rsi_indicator(df, params, limit, price_source):
    """RSI 지표를 계산하는 함수"""
    length = params.get('length', 14)
    
    try:
        calculated_indicator = compute_rsi(df, length=length, price_source=price_source)
        if calculated_indicator is None or calculated_indicator.empty:
            return None, f"RSI 계산 결과가 비어 있습니다. (length: {length})"
            
        processed_series = process_indicator_series(calculated_indicator, limit)
        return processed_series, None
    except Exception as e:
        return None, f"RSI 계산 중 오류 발생: {str(e)}"

def calculate_sma_indicator(df, params, limit, price_source):
    """SMA 지표를 계산하는 함수"""
    length = params.get('length', 20)
    
    try:
        calculated_indicator = compute_sma(df, length=length, price_source=price_source)
        if calculated_indicator is None or calculated_indicator.empty:
            return None, f"SMA 계산 결과가 비어 있습니다. (length: {length})"
            
        processed_series = process_indicator_series(calculated_indicator, limit)
        return processed_series, None
    except Exception as e:
        return None, f"SMA 계산 중 오류 발생: {str(e)}"

def calculate_ema_indicator(df, params, limit, price_source):
    """EMA 지표를 계산하는 함수"""
    length = params.get('length', 20)
    
    try:
        calculated_indicator = compute_ema(df, length=length, price_source=price_source)
        if calculated_indicator is None or calculated_indicator.empty:
            return None, f"EMA 계산 결과가 비어 있습니다. (length: {length})"
            
        processed_series = process_indicator_series(calculated_indicator, limit)
        return processed_series, None
    except Exception as e:
        return None, f"EMA 계산 중 오류 발생: {str(e)}"

def calculate_macd_indicator(df, params, limit, price_source):
    """MACD 지표를 계산하는 함수"""
    fast_length = params.get('fast', 12)
    slow_length = params.get('slow', 26)
    signal_length = params.get('signal', 9)
    
    try:
        macd_result = compute_macd(
            df, 
            fast_length=fast_length, 
            slow_length=slow_length, 
            signal_length=signal_length, 
            price_source=price_source
        )
        
        if macd_result is None:
            return None, f"MACD 계산 결과가 비어 있습니다."
            
        macd_line, signal_line, histogram = macd_result
        
        # 각 시리즈 처리
        processed_macd = process_indicator_series(macd_line, limit)
        processed_signal = process_indicator_series(signal_line, limit)
        processed_hist = process_indicator_series(histogram, limit)
        
        return (processed_macd, processed_signal, processed_hist), None
    except Exception as e:
        return None, f"MACD 계산 중 오류 발생: {str(e)}"

def calculate_bbands_indicator(df, params, limit, price_source):
    """볼린저밴드 지표를 계산하는 함수"""
    length = params.get('length', 20)
    std_dev = params.get('std', 2.0)
    
    try:
        bbands_result = compute_bbands(
            df, 
            length=length, 
            std_dev=std_dev, 
            price_source=price_source
        )
        
        if bbands_result is None:
            return None, f"볼린저밴드 계산 결과가 비어 있습니다."
            
        lower_band, middle_band, upper_band = bbands_result
        
        # 각 시리즈 처리
        processed_lower = process_indicator_series(lower_band, limit)
        processed_middle = process_indicator_series(middle_band, limit)
        processed_upper = process_indicator_series(upper_band, limit)
        
        return (processed_lower, processed_middle, processed_upper), None
    except Exception as e:
        return None, f"볼린저밴드 계산 중 오류 발생: {str(e)}"

def calculate_stochastic_indicator(df, params, limit, price_source_high, price_source_low, price_source_close):
    """Stochastic Oscillator 지표를 계산하는 함수"""
    k_period = params.get('k_period', 14)
    d_period = params.get('d_period', 3)
    smooth_k = params.get('smooth_k', 3)
    
    try:
        stoch_result = compute_stochastic_oscillator(
            df,
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
            price_source_high=price_source_high,
            price_source_low=price_source_low,
            price_source_close=price_source_close
        )
        
        if stoch_result is None:
            return None, f"Stochastic Oscillator 계산 결과가 비어 있습니다."
            
        percent_k, percent_d = stoch_result
        
        processed_k = process_indicator_series(percent_k, limit)
        processed_d = process_indicator_series(percent_d, limit)
        
        return (processed_k, processed_d), None
    except Exception as e:
        return None, f"Stochastic Oscillator 계산 중 오류 발생: {str(e)}"

def calculate_atr_indicator(df, params, limit, price_source_high, price_source_low, price_source_close):
    """ATR 지표를 계산하는 함수"""
    period = params.get('period', 14)
    
    try:
        calculated_indicator = compute_atr(
            df, 
            period=period, 
            price_source_high=price_source_high,
            price_source_low=price_source_low,
            price_source_close=price_source_close
        )
        if calculated_indicator is None or calculated_indicator.empty:
            return None, f"ATR 계산 결과가 비어 있습니다. (period: {period})"
            
        processed_series = process_indicator_series(calculated_indicator, limit)
        return processed_series, None
    except Exception as e:
        return None, f"ATR 계산 중 오류 발생: {str(e)}"

@mcp.tool(
    name="calculate_technical_indicator",
    description="Fetches OHLCV data for a given symbol and timeframe, then calculates a specified technical indicator "
                "(e.g., RSI, SMA, EMA, MACD, Bollinger Bands, Stochastic Oscillator, ATR). Returns a time series of calculated indicator values. "
                "The number of data points returned corresponds to the OHLCV data fetched (controlled by 'ohlcv_limit' in indicator_params).",
    tags={"market_data", "technical_analysis", "indicator", "charting", "RSI", "SMA", "EMA", "MACD", "BBANDS", "STOCH", "ATR"}
)
async def calculate_technical_indicator_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'upbit'). Case-insensitive.")],
    symbol: Annotated[str, Field(description="The trading symbol to calculate the indicator for (e.g., 'BTC/USDT', 'ETH/KRW').")],
    timeframe: Annotated[TimeframeLiteral, Field(description="The candle timeframe for OHLCV data. Common supported values are provided. Always check the specific exchange's documentation for their full list of supported timeframes as it can vary.")],
    indicator_name: Annotated[Literal["RSI", "SMA", "EMA", "MACD", "BBANDS", "STOCH", "ATR"], 
                            Field(description="The name of the technical indicator to calculate. Supported: RSI, SMA, EMA, MACD, BBANDS, STOCH, ATR.")],
    ohlcv_limit: Annotated[Optional[int], Field(description="Optional: The number of OHLCV data points to fetch. Default is 50. Check exchange for default and maximum limits.", gt=0)] = None,
    indicator_params: Annotated[Optional[str], Field(
        description='''Optional: A JSON string representing a dictionary of parameters for the chosen indicator. All parameters within the dictionary are optional and have defaults.
        Example JSON string for RSI: {"length": 14, "price_source": "close"}.
        Parameter details for the dictionary:
        For RSI: {'length': 14, 'price_source': 'close'}.
        For SMA/EMA: {'length': 20, 'price_source': 'close'}.
        For MACD: {'fast': 12, 'slow': 26, 'signal': 9, 'price_source': 'close'}.
        For BBANDS (Bollinger Bands): {'length': 20, 'std': 2.0, 'price_source': 'close'}.
        For STOCH (Stochastic Oscillator): {'k_period': 14, 'd_period': 3, 'smooth_k': 3, 'price_source_high': 'high', 'price_source_low': 'low', 'price_source_close': 'close'}.
        For ATR (Average True Range): {'period': 14, 'price_source_high': 'high', 'price_source_low': 'low', 'price_source_close': 'close'}.
        Valid 'price_source' values for single-price indicators: 'open', 'high', 'low', 'close' (default), 'hlc3', 'ohlc4'.
        Ensure the JSON string is correctly formatted.'''
    )] = None,
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. If not provided, the system may use pre-configured credentials or proceed unauthenticated. If authentication is used (with directly provided or pre-configured keys), it may offer benefits like enhanced access or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange. Used with an API key if authentication is performed (whether keys are provided directly or pre-configured).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange. Used with an API key if authentication is performed and the exchange requires it (whether keys are provided directly or pre-configured).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for CCXT client instantiation when fetching OHLCV data, "
                                                        "e.g., `{'options': {'defaultType': 'future'}}` if fetching for non-spot markets like futures.")] = None
) -> Dict:
    """기술적 지표를 계산하는 함수 (리팩토링 버전)"""
    
    # 1. 파라미터 파싱 및 검증
    parsed_params = parse_indicator_params(indicator_params)
    
    # 요청된 최종 데이터 길이 결정
    requested_final_length = ohlcv_limit or parsed_params.get('ohlcv_limit', 50)
    if not isinstance(requested_final_length, int) or requested_final_length <= 0:
        requested_final_length = 50
    
    # 실제 OHLCV 데이터 가져올 길이 (계산용 버퍼 추가)
    ohlcv_fetch_limit = requested_final_length + 100
    
    # 가격 소스 컬럼 결정 (STOCH는 별도 처리)
    price_source_col = None
    price_source_high_col = None
    price_source_low_col = None
    price_source_close_col = None

    multi_source_indicators = ["STOCH", "ATR"]
    if indicator_name not in multi_source_indicators:
        price_source_col = parsed_params.get('price_source', 'close').lower()
        valid_price_sources = ['open', 'high', 'low', 'close', 'hlc3', 'ohlc4']
        if price_source_col not in valid_price_sources:
            return {"error": f"잘못된 'price_source': {price_source_col}. 다음 중 하나여야 합니다: {valid_price_sources}."}
    else: # STOCH and ATR use specific high, low, close sources
        price_source_high_col = parsed_params.get('price_source_high', 'high').lower()
        price_source_low_col = parsed_params.get('price_source_low', 'low').lower()
        price_source_close_col = parsed_params.get('price_source_close', 'close').lower()
        # Validation for these columns will happen in prepare_dataframe or the specific compute function.

    # 2. OHLCV 데이터 가져오기
    ohlcv_data, fetch_error = await fetch_ohlcv_data(
        exchange_id, symbol, timeframe, ohlcv_fetch_limit, 
        api_key, secret_key, passphrase, params
    )
    
    if fetch_error:
        return {"error": fetch_error}
    
    # 3. 데이터프레임 준비
    # For STOCH or ATR, price_source_col is not used directly by prepare_dataframe for selecting the final calculation column,
    # but it needs one of the price columns (e.g., 'close') to check for all NaNs initially during dataframe preparation.
    initial_check_price_source = price_source_close_col if indicator_name in multi_source_indicators else price_source_col
    df, df_error = prepare_dataframe(ohlcv_data, initial_check_price_source)
    if df_error:
        return {"error": df_error}
    
    # 4. 실제 파라미터 사용값 기록 (결과에 포함시키기 위함)
    actual_params_used = parsed_params.copy()
    actual_params_used['ohlcv_limit'] = requested_final_length
    
    price_source_display = {}
    if indicator_name in multi_source_indicators:
        price_source_display = {
            'high': price_source_high_col,
            'low': price_source_low_col,
            'close': price_source_close_col
        }
        actual_params_used.update({
            'price_source_high': price_source_high_col,
            'price_source_low': price_source_low_col,
            'price_source_close': price_source_close_col
        })
        # Remove single 'price_source' if it was in parsed_params for multi-source indicators, as it's not used.
        actual_params_used.pop('price_source', None)
    else:
        price_source_display = price_source_col
        actual_params_used['price_source'] = price_source_col

    # 5. 지표 계산 및 결과 생성
    indicator_output = {
        "indicator_name": indicator_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "params_used": actual_params_used,
        "price_source_used": price_source_display,
        "data": []
    }
    
    try:
        if indicator_name == "RSI":
            length = parsed_params.get('length', 14)
            actual_params_used['length'] = length
            
            result, error = calculate_rsi_indicator(df, parsed_params, requested_final_length, price_source_col)
            if error:
                return {"error": error}
            indicator_output["data"] = format_series_to_list(result, "value")
            
        elif indicator_name == "SMA":
            length = parsed_params.get('length', 20)
            actual_params_used['length'] = length
            
            result, error = calculate_sma_indicator(df, parsed_params, requested_final_length, price_source_col)
            if error:
                return {"error": error}
            indicator_output["data"] = format_series_to_list(result, "value")
            
        elif indicator_name == "EMA":
            length = parsed_params.get('length', 20)
            actual_params_used['length'] = length
            
            result, error = calculate_ema_indicator(df, parsed_params, requested_final_length, price_source_col)
            if error:
                return {"error": error}
            indicator_output["data"] = format_series_to_list(result, "value")
            
        elif indicator_name == "MACD":
            fast_length = parsed_params.get('fast', 12)
            slow_length = parsed_params.get('slow', 26)
            signal_length = parsed_params.get('signal', 9)
            actual_params_used.update({'fast': fast_length, 'slow': slow_length, 'signal': signal_length})
            
            result, error = calculate_macd_indicator(df, parsed_params, requested_final_length, price_source_col)
            if error:
                return {"error": error}
            macd_series, signal_series, hist_series = result
            indicator_output["data"] = format_macd_to_list(macd_series, signal_series, hist_series)
            
        elif indicator_name == "BBANDS":
            length = parsed_params.get('length', 20)
            std_dev = parsed_params.get('std', 2.0)
            actual_params_used.update({'length': length, 'std': std_dev})
            
            result, error = calculate_bbands_indicator(df, parsed_params, requested_final_length, price_source_col)
            if error:
                return {"error": error}
            lower_band, middle_band, upper_band = result
            indicator_output["data"] = format_bbands_to_list(lower_band, middle_band, upper_band)

        elif indicator_name == "STOCH":
            k_period = parsed_params.get('k_period', 14)
            d_period = parsed_params.get('d_period', 3)
            smooth_k = parsed_params.get('smooth_k', 3)
            actual_params_used.update({'k_period': k_period, 'd_period': d_period, 'smooth_k': smooth_k})

            result, error = calculate_stochastic_indicator(
                df, parsed_params, requested_final_length, 
                price_source_high_col, price_source_low_col, price_source_close_col
            )
            if error:
                return {"error": error}
            percent_k, percent_d = result
            indicator_output["data"] = format_stochastic_to_list(percent_k, percent_d)

        elif indicator_name == "ATR":
            period = parsed_params.get('period', 14)
            actual_params_used['period'] = period

            result, error = calculate_atr_indicator(
                df, parsed_params, requested_final_length,
                price_source_high_col, price_source_low_col, price_source_close_col
            )
            if error:
                return {"error": error}
            indicator_output["data"] = format_series_to_list(result, "value") # ATR is a single series
            
        else:
            return {"error": f"지표 '{indicator_name}'는 현재 지원되지 않습니다."}
            
        # 데이터가 비어있을 경우 빈 리스트 반환
        if indicator_output["data"] is None:
            indicator_output["data"] = []
            
        return indicator_output
        
    except Exception as e:
        # 서버 로그에 자세한 오류 기록
        print(f"calculate_technical_indicator_tool에서 예상치 못한 오류 발생 - {indicator_name}/{symbol}: {e}")
        return {"error": f"{symbol}에 대한 {indicator_name} 계산 중 오류가 발생했습니다."}

# --- Main execution (for running the server) ---
if __name__ == "__main__":
    print("Starting CCXT MCP Server (Async with Annotated Params and Tool Metadata)...")
    mcp.run(transport="stdio")