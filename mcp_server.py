# mcp_server.py

from typing import Optional, List, Union, Dict, Annotated, Literal
import ccxt.async_support as ccxtasync # Changed for asynchronous support and alias
from fastmcp import FastMCP
import asyncio
from pydantic import Field

# Initialize FastMCP
mcp = FastMCP("CCXT MCP Server 🚀")

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
    api_key: Annotated[Optional[str], Field(description="Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange (e.g., for KuCoin, OKX).")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key with withdrawal permissions.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the API.")] = None,
    tag: Annotated[Optional[str], Field(description="Optional: Destination tag, memo, or payment ID for certain currencies (e.g., XRP, XLM, EOS). Check exchange/currency requirements.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for withdrawals.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError, ccxtasync.InsufficientFunds) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the exchange.")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional/Required: The symbol (e.g., 'BTC/USDT:USDT' for futures, 'BTC/USDT' for margin) to set leverage for. "
                                                       "Some exchanges require it, others set it account-wide or per market type. Check exchange documentation.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange.")] = None,
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
    except ccxtasync.NotSupported as e:
        return {"error": f"Exchange '{exchange_id}' does not support the unified setLeverage with the provided arguments. Error: {str(e)}. You may need to use exchange-specific 'params' or check symbol requirements."}
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.ExchangeError, ValueError) as e:
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
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange. May provide access to more data or higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase, if required by the exchange.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key for the exchange.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key for the exchange.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Optional: Your API key. May provide higher rate limits.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Optional: Your secret key.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key with trading permissions.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the API.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading.")] = None,
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
            return {"error": f"Exchange '{exchange_id}' does not support createOrder."}
        
        order = await exchange.createOrder(symbol, 'limit', side, amount, price, params=tool_params)
        return order
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError, ccxtasync.InsufficientFunds, ccxtasync.InvalidOrder) as e:
        return {"error": str(e)}
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
    amount: Annotated[float, Field(description="The quantity of the base currency to trade (for a market buy) or the quantity to sell (for a market sell). Must be greater than 0. "
                                            "For market buy orders on certain exchanges like Upbit, this 'amount' should be the total cost (quote currency) to spend, and you MUST pass `{'createMarketBuyOrderRequiresPrice': False}` in the `params` argument. "
                                            "For other exchanges, `params` might support `{'quoteOrderQty': quote_amount}` to specify the amount in quote currency.", gt=0)],
    api_key: Annotated[Optional[str], Field(description="Your API key with trading permissions.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the API.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the CCXT `createOrder` call. "
                                                        "Common uses include `{'clientOrderId': 'your_custom_id'}`. "
                                                        "For market buy orders, some exchanges allow `{'quoteOrderQty': quote_amount}` to specify the amount in quote currency (e.g., spend 100 USDT on BTC). "
                                                        "Example: `{'clientOrderId': 'my_market_buy_001', 'quoteOrderQty': 100}` for a $100 market buy. "
                                                        "For Upbit market buy by cost: `{'createMarketBuyOrderRequiresPrice': False}` (and provide total cost in 'amount'). "
                                                        "No `options` for client instantiation are typically needed for spot orders.")] = None
) -> Dict:
    """Internal use: Creates a spot market order. Primary description is in @mcp.tool decorator."""
    if not api_key or not secret_key:
        return {"error": "API key and secret key are required for create_spot_market_order."}
        
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=api_key_info_dict, exchange_config_options=client_config_options)
        if not exchange.has['createOrder']:
            return {"error": f"Exchange '{exchange_id}' does not support createOrder."}
        
        order = await exchange.createOrder(symbol, 'market', side, amount, params=tool_params)
        return order
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError, ccxtasync.InsufficientFunds, ccxtasync.InvalidOrder) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key with trading permissions.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the API.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError, ccxtasync.InsufficientFunds, ccxtasync.InvalidOrder) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key with trading permissions.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the API.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange for trading.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError, ccxtasync.InsufficientFunds, ccxtasync.InvalidOrder) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key with trading permissions.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key for the API.")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional/Required: The symbol of the order (e.g., 'BTC/USDT', 'BTC/USDT:USDT'). "
                                                       "Required by some exchanges for `cancelOrder`, optional for others. Check exchange documentation.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError, ccxtasync.OrderNotFound) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key.")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional: The symbol (e.g., 'BTC/USDT', 'ETH/USDT:USDT') to fetch orders for. If omitted, orders for all symbols may be returned (exchange-dependent).")] = None,
    since: Annotated[Optional[int], Field(description="Optional: Timestamp in milliseconds (UTC epoch) to fetch orders created since this time.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of orders to retrieve. Check exchange for default and maximum limits.", gt=0)] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
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
    api_key: Annotated[Optional[str], Field(description="Your API key.")] = None,
    secret_key: Annotated[Optional[str], Field(description="Your secret key.")] = None,
    symbol: Annotated[Optional[str], Field(description="Optional: The symbol (e.g., 'BTC/USDT', 'BTC/USDT:USDT') to fetch your trades for. If omitted, trades for all symbols may be returned (exchange-dependent).")] = None,
    since: Annotated[Optional[int], Field(description="Optional: Timestamp in milliseconds (UTC epoch) to fetch trades executed since this time.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of trades to retrieve. Check exchange for default and maximum limits.", gt=0)] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange.")] = None,
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
    except (ccxtasync.NetworkError, ccxtasync.AuthenticationError, ccxtasync.ExchangeNotFound, ccxtasync.NotSupported, ccxtasync.ExchangeError, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred in fetch_my_trade_history: {str(e)}"}
    finally:
        if exchange:
            await exchange.close()

# --- Main execution (for running the server) ---
if __name__ == "__main__":
    print("Starting CCXT MCP Server (Async with Annotated Params and Tool Metadata)...")
    mcp.run()