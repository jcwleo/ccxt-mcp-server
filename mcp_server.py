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
    CCXT 거래소 인스턴스를 가져오거나 생성합니다 (캐싱 포함 - 현재 구현에는 캐싱 로직 없음)
    API 키 정보가 제공되면 인증된 인스턴스를 생성/반환합니다
    키 정보가 없으면 인증되지 않은 인스턴스를 반환합니다 (공용 API 호출용)
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

@mcp.tool(
    name="fetch_account_balance",
    description="Fetches the current balance of an account from a specified cryptocurrency exchange. Requires API authentication.",
    tags={"account", "balance", "private", "spot", "margin", "futures"}
)
async def fetch_balance_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'binance', 'upbit').")],
    api_key: Annotated[str, Field(description="Your API key for the exchange.")],
    secret_key: Annotated[str, Field(description="Your secret key for the exchange.")],
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase (e.g., for KuCoin, OKX).")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the exchange API. Can include 'options' for CCXT client config.")] = None
) -> Dict:
    """Internal use: Fetches account balance. Primary description is in @mcp.tool decorator."""
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
    description="Fetches the deposit address for a specific cryptocurrency on a given exchange. Requires API authentication.",
    tags={"account", "deposit", "address", "private"}
)
async def fetch_deposit_address_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="Your API key for the exchange.")],
    secret_key: Annotated[str, Field(description="Your secret key for the exchange.")],
    code: Annotated[str, Field(description="Currency code (e.g., 'BTC', 'ETH').")],
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the exchange API. Can include 'options' for CCXT client config.")] = None
) -> Dict:
    """Internal use: Fetches deposit address. Primary description is in @mcp.tool decorator."""
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
    description="Withdraws a specified amount of cryptocurrency to a given address. Requires API authentication and withdrawal permissions.",
    tags={"account", "withdrawal", "transaction", "private"}
)
async def withdraw_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="API key with withdrawal permissions.")],
    secret_key: Annotated[str, Field(description="Secret key for the API.")],
    code: Annotated[str, Field(description="Currency code for withdrawal (e.g., 'BTC', 'ETH').")],
    amount: Annotated[float, Field(description="Amount of currency to withdraw.", gt=0)],
    address: Annotated[str, Field(description="Destination address for the withdrawal.")],
    tag: Annotated[Optional[str], Field(description="Optional: Destination tag/memo for certain currencies (e.g., XRP, XLM).")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required by the exchange.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the exchange API. Can include 'options' for CCXT client config.")] = None
) -> Dict:
    """Internal use: Withdraws cryptocurrency. Primary description is in @mcp.tool decorator."""
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
    description="Fetches open positions for futures or derivatives trading from an exchange. Requires API authentication.",
    tags={"account", "positions", "futures", "derivatives", "private"}
)
async def fetch_positions_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (must support futures).")],
    api_key: Annotated[str, Field(description="Your API key for the exchange.")],
    secret_key: Annotated[str, Field(description="Your secret key for the exchange.")],
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Crucial for specifying market type (e.g., {'options': {'defaultType': 'future'}}) for client instantiation.")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches open positions. Primary description is in @mcp.tool decorator."""
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
    description="Sets the leverage for a specific trading symbol (usually for futures markets). Requires API authentication.",
    tags={"trading", "leverage", "futures", "private"}
)
async def set_leverage_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="Your API key for the exchange.")],
    secret_key: Annotated[str, Field(description="Your secret key for the exchange.")],
    leverage: Annotated[int, Field(description="The desired leverage (e.g., 10 for 10x).", gt=0)],
    symbol: Annotated[Optional[str], Field(description="Optional/Required: The symbol (e.g., 'BTC/USDT') to set leverage for. Check exchange documentation.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: Your API passphrase.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Crucial for market type (e.g., {'options': {'defaultType': 'future'}}) for client instantiation and for exchange-specific calls.")] = None
) -> Dict:
    """Internal use: Sets trading leverage. Primary description is in @mcp.tool decorator."""
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
    description="Fetches historical Open-High-Low-Close-Volume (OHLCV) candlestick data for a symbol. Typically a public endpoint.",
    tags={"market_data", "ohlcv", "candlestick", "historical_data", "public"}
)
async def fetch_ohlcv_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    symbol: Annotated[str, Field(description="The trading symbol to fetch OHLCV data for (e.g., 'BTC/USDT').")],
    timeframe: Annotated[str, Field(description="The length of time each candle represents (e.g., '1m', '5m', '1h', '1d').")],
    since: Annotated[Optional[int], Field(description="Optional: The earliest time in milliseconds (UTC) to fetch OHLCV data from (e.g., 1502962800000).", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: The maximum number of OHLCV candles to return.", gt=0)] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the exchange API. Can include {'options': ...} for client instantiation or {'price': 'mark'/'index'} for specific OHLCV types.")] = None
) -> Union[List[List[Union[int, float]]], Dict]:
    """Internal use: Fetches OHLCV data. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=None, exchange_config_options=client_config_options)
        if not exchange.has['fetchOHLCV']:
            return {"error": f"Exchange '{exchange_id}' does not support fetchOHLCV."}
        
        # The 'price' parameter for mark/index price OHLCV is passed within the 'params' argument to fetchOHLCV itself.
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
    description="Fetches the current funding rate for a perpetual futures contract symbol. Typically a public endpoint.",
    tags={"market_data", "funding_rate", "futures", "public"}
)
async def fetch_funding_rate_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    symbol: Annotated[str, Field(description="The symbol to fetch the funding rate for (e.g., 'BTC/USDT:USDT').")],
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Market type might be needed via {'options': {'defaultType': 'future'}} for client instantiation.")] = None
) -> Dict:
    """Internal use: Fetches funding rate. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=None, exchange_config_options=client_config_options)
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
    description="Fetches the long/short ratio for a symbol, often for futures markets, by calling specific exchange API methods. Typically a public endpoint.",
    tags={"market_data", "sentiment", "long_short_ratio", "futures", "public"}
)
async def fetch_long_short_ratio_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    symbol: Annotated[str, Field(description="Symbol (e.g., 'BTC/USDT').")],
    timeframe: Annotated[str, Field(description="Timeframe (e.g., '5m', '1h').")],
    limit: Annotated[Optional[int], Field(description="Optional: Number of data points.", gt=0)] = None,
    params: Annotated[Optional[Dict], Field(description="Crucial. Use 'method_name' for the CCXT implicit method and 'method_params' for its arguments. Can also include {'options': {'defaultType': 'future'}} for client instantiation.")] = None
) -> Dict:
    """Internal use: Fetches long/short ratio. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=None, exchange_config_options=client_config_options)
        
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
    description="Fetches market data for a specific options contract, usually by using the exchange's ticker functionality. Typically a public endpoint.",
    tags={"market_data", "options", "ticker", "public"}
)
async def fetch_option_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange (e.g., 'deribit').")],
    symbol: Annotated[str, Field(description="The specific option symbol (e.g., 'BTC-28JUN24-70000-C').")],
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Market type via {'options': {'defaultType': 'option'}} might be needed for client instantiation.")] = None
) -> Dict:
    """Internal use: Fetches option contract data. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=None, exchange_config_options=client_config_options)

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
    description="Fetches the latest price ticker data (bid, ask, last price, volume, etc.) for a trading symbol. Typically a public endpoint.",
    tags={"market_data", "ticker", "price", "public"}
)
async def fetch_ticker_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    symbol: Annotated[str, Field(description="The symbol to fetch ticker for (e.g., 'BTC/USDT').")],
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the exchange API. Can include {'options': ...} for client instantiation.")] = None
) -> Dict:
    """Internal use: Fetches market ticker. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=None, exchange_config_options=client_config_options)
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
    description="Fetches recent publicly executed trades for a specific trading symbol. Typically a public endpoint.",
    tags={"market_data", "trades", "history", "public"}
)
async def fetch_trades_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    symbol: Annotated[str, Field(description="The symbol to fetch trades for.")],
    since: Annotated[Optional[int], Field(description="Optional: Timestamp (milliseconds, UTC) to fetch trades since.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of trades to fetch.", gt=0)] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters for the exchange API. Can include {'options': ...} for client instantiation.")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches public market trades. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    client_config_options = tool_params.pop('options', None)
    exchange : ccxtasync.Exchange = None
    try:
        exchange = await get_exchange_instance(exchange_id, api_key_info=None, exchange_config_options=client_config_options)
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
    description="Places a new spot limit order on the exchange. Requires API authentication and trading permissions.",
    tags={"trading", "order", "create", "spot", "limit", "private"}
)
async def create_spot_limit_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="API key with trading permissions.")],
    secret_key: Annotated[str, Field(description="Secret key for the API.")],
    symbol: Annotated[str, Field(description="Symbol to trade (e.g., 'BTC/USDT').")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' or 'sell'.")],
    amount: Annotated[float, Field(description="Amount of currency to trade.", gt=0)],
    price: Annotated[float, Field(description="Price for the limit order.", gt=0)],
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters (e.g., clientOrderId). Can include 'options' for CCXT client config (though typically not needed for spot). ")] = None
) -> Dict:
    """Internal use: Creates a spot limit order. Primary description is in @mcp.tool decorator."""
    tool_params = params.copy() if params else {}
    api_key_info_dict = {'apiKey': api_key, 'secret': secret_key}
    if passphrase:
        api_key_info_dict['password'] = passphrase

    client_config_options = tool_params.pop('options', None) # e.g., if exchange needs specific options even for spot
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
    description="Places a new spot market order on the exchange. Requires API authentication and trading permissions.",
    tags={"trading", "order", "create", "spot", "market", "private"}
)
async def create_spot_market_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="API key with trading permissions.")],
    secret_key: Annotated[str, Field(description="Secret key for the API.")],
    symbol: Annotated[str, Field(description="Symbol to trade (e.g., 'BTC/USDT').")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' or 'sell'.")],
    amount: Annotated[float, Field(description="Amount of currency to trade.", gt=0)],
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters (e.g., clientOrderId). Can include 'options' for CCXT client config (though typically not needed for spot). ")] = None
) -> Dict:
    """Internal use: Creates a spot market order. Primary description is in @mcp.tool decorator."""
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
        
        # Market orders do not have a price parameter for createOrder
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
    description="Places a new futures limit order. Requires API authentication. Ensure client is configured for futures (e.g., params={'options': {'defaultType': 'future'}}).",
    tags={"trading", "order", "create", "futures", "limit", "private"}
)
async def create_futures_limit_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="API key with trading permissions.")],
    secret_key: Annotated[str, Field(description="Secret key for the API.")],
    symbol: Annotated[str, Field(description="Futures symbol to trade (e.g., 'BTC/USDT:USDT').")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' or 'sell'.")],
    amount: Annotated[float, Field(description="Amount of contracts/currency to trade.", gt=0)],
    price: Annotated[float, Field(description="Price for the limit order.", gt=0)],
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. CRITICAL: Include {'options': {'defaultType': 'future'}} or similar for client instantiation if not default for the exchange.")] = None
) -> Dict:
    """Internal use: Creates a futures limit order. Primary description is in @mcp.tool decorator."""
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
    description="Places a new futures market order. Requires API authentication. Ensure client is configured for futures (e.g., params={'options': {'defaultType': 'future'}}).",
    tags={"trading", "order", "create", "futures", "market", "private"}
)
async def create_futures_market_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="API key with trading permissions.")],
    secret_key: Annotated[str, Field(description="Secret key for the API.")],
    symbol: Annotated[str, Field(description="Futures symbol to trade (e.g., 'BTC/USDT:USDT').")],
    side: Annotated[Literal["buy", "sell"], Field(description="Order side: 'buy' or 'sell'.")],
    amount: Annotated[float, Field(description="Amount of contracts/currency to trade.", gt=0)],
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. CRITICAL: Include {'options': {'defaultType': 'future'}} or similar for client instantiation if not default for the exchange.")] = None
) -> Dict:
    """Internal use: Creates a futures market order. Primary description is in @mcp.tool decorator."""
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
        
        order = await exchange.createOrder(symbol, 'market', side, amount, params=tool_params) # Market orders don't take price
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
    description="Cancels an existing open order on the exchange. Requires API authentication.",
    tags={"trading", "order", "cancel", "private"}
)
async def cancel_order_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="Your API key.")],
    secret_key: Annotated[str, Field(description="Your secret key.")],
    id: Annotated[str, Field(description="The order ID to cancel.")],
    symbol: Annotated[Optional[str], Field(description="Optional: Symbol of the order (e.g., 'BTC/USDT'). Required by some exchanges.")] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Can include 'options' for CCXT client config.")] = None
) -> Dict:
    """Internal use: Cancels an order. Primary description is in @mcp.tool decorator."""
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
    description="Fetches a list of orders (open, closed, or all) for an account or a specific symbol. Requires API authentication.",
    tags={"account", "orders", "history", "private"}
)
async def fetch_orders_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="Your API key.")],
    secret_key: Annotated[str, Field(description="Your secret key.")],
    symbol: Annotated[Optional[str], Field(description="Optional: Symbol (e.g., 'BTC/USDT') to fetch orders for.")] = None,
    since: Annotated[Optional[int], Field(description="Optional: Timestamp (milliseconds, UTC) to fetch orders since.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of orders to fetch.", gt=0)] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Can include 'options' for CCXT client config.")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches order history. Primary description is in @mcp.tool decorator."""
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
    description="Fetches the history of trades executed by the authenticated user. Requires API authentication.",
    tags={"account", "trades", "history", "private"}
)
async def fetch_my_trades_tool(
    exchange_id: Annotated[str, Field(description="The ID of the exchange.")],
    api_key: Annotated[str, Field(description="Your API key.")],
    secret_key: Annotated[str, Field(description="Your secret key.")],
    symbol: Annotated[Optional[str], Field(description="Optional: Symbol (e.g., 'BTC/USDT') to fetch your trades for.")] = None,
    since: Annotated[Optional[int], Field(description="Optional: Timestamp (milliseconds, UTC) to fetch your trades since.", ge=0)] = None,
    limit: Annotated[Optional[int], Field(description="Optional: Maximum number of your trades to fetch.", gt=0)] = None,
    passphrase: Annotated[Optional[str], Field(description="Optional: API passphrase if required.")] = None,
    params: Annotated[Optional[Dict], Field(description="Optional: Extra parameters. Can include 'options' for CCXT client config.")] = None
) -> Union[List[Dict], Dict]:
    """Internal use: Fetches user's trade history. Primary description is in @mcp.tool decorator."""
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