# CCXT MCP Server
[![smithery badge](https://smithery.ai/badge/@jcwleo/ccxt-mcp-server)](https://smithery.ai/server/@jcwleo/ccxt-mcp-server)

This project provides a Model Context Protocol (MCP) server that exposes various functions from the [CCXT](https://github.com/ccxt/ccxt) library as tools for Large Language Models (LLMs).

It allows LLMs to interact with cryptocurrency exchanges for tasks like fetching balances, market data, creating orders, and more, in a standardized and asynchronous way.

This server is built using [FastMCP](https://gofastmcp.com/), which simplifies the process of creating MCP servers in Python.

## Features

*   **CCXT Integration**: Wraps common CCXT functions for exchange interaction.
*   **Asynchronous**: Built using `asyncio` and `ccxt.async_support` for efficient non-blocking operations.
*   **Clear Tool Definitions**: Uses `typing.Annotated` and `pydantic.Field` for clear parameter descriptions and constraints, making it easier for LLMs (and developers) to understand and use the tools.
*   **Authentication Handling**: Supports API key, secret, and passphrase authentication for private endpoints.
*   **Public & Private Tools**: Provides separate tools for public market data and private account actions.

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/jcwleo/ccxt-mcp-server.git
    cd ccxt-mcp-server
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The required libraries are listed in `requirements.txt`. You can install them using `pip` or `uv`.

    *   **Using `pip`:**
        ```bash
        pip install -r requirements.txt
        ```
    *   **Using `uv` (if installed):**
        ```bash
        uv pip install -r requirements.txt
        # Or, if you prefer uv's environment management:
        # uv sync
        ```

## Running the Server

Once the dependencies are installed, you can run the MCP server directly:

```bash
uv run mcp_server.py
```

You should see output indicating the server has started, similar to:

```
Starting CCXT MCP Server (Async with Annotated Params and Tool Metadata)...
# ... (FastMCP server startup logs)
```

The server will then be available for MCP clients to connect to (typically on a default port managed by FastMCP, unless configured otherwise).

## MCP Server Configuration (for MCP Clients)

If you are using an MCP client that requires manual server configuration (like the Claude desktop app), you'll need to provide a configuration similar to the following JSON.

Create a `claude_desktop_config.json` file (or the equivalent for your MCP client) with the following structure:

```json
{
  "mcpServers": {
    "ccxt-mcp-server": {
      "command": "npx",
      "args": ["mcp-remote", "http://127.0.0.1:8000/mcp"]
    }
  }
}
```

This configuration tells your MCP client how to start and communicate with the CCXT MCP server.

## Available MCP Tools

This server exposes the following tools, categorized by whether they require API authentication.

### Tools Requiring API Authentication (Private)

*   **`fetch_account_balance`**: Fetches the current account balance.
*   **`fetch_deposit_address`**: Fetches the deposit address for a currency.
*   **`withdraw_cryptocurrency`**: Withdraws cryptocurrency to a specified address.
*   **`fetch_open_positions`**: Fetches open positions (primarily for futures/derivatives).
*   **`set_trading_leverage`**: Sets leverage for a trading symbol (primarily for futures).
*   **`create_spot_limit_order`**: Places a new spot limit order.
*   **`create_spot_market_order`**: Places a new spot market order.
*   **`create_futures_limit_order`**: Places a new futures limit order.
*   **`create_futures_market_order`**: Places a new futures market order.
*   **`cancel_order`**: Cancels an existing open order.
*   **`fetch_order_history`**: Fetches the history of orders (open/closed).
*   **`fetch_my_trade_history`**: Fetches the history of trades executed by the user.

### Tools for Public Data (No Authentication Required)

*   **`fetch_ohlcv`**: Fetches historical OHLCV (candlestick) data.
*   **`fetch_funding_rate`**: Fetches the funding rate for a perpetual futures contract.
*   **`fetch_long_short_ratio`**: Fetches the long/short ratio (requires exchange-specific `params`).
*   **`fetch_option_contract_data`**: Fetches market data for an options contract.
*   **`fetch_market_ticker`**: Fetches the latest price ticker data for a symbol.
*   **`fetch_public_market_trades`**: Fetches recent public trades for a symbol.
*   **`calculate_technical_indicator_tool`**: Fetches OHLCV data and calculates a specified technical indicator (e.g., RSI, SMA, EMA, MACD, Bollinger Bands, Stochastic Oscillator, Average True Range (ATR)).

Each tool has detailed parameter descriptions available via the MCP protocol itself, thanks to the use of `Annotated` and `pydantic.Field`.

## Usage Notes

*   **Futures/Options**: When using tools related to futures or options (e.g., `fetch_open_positions`, `create_futures_limit_order`, `fetch_funding_rate`), ensure you correctly configure the CCXT client via the `params` argument, specifically passing `{'options': {'defaultType': 'future'}}` (or `'swap'`, `'option'` as needed) if the exchange requires it or doesn't default to the desired market type.
*   **`fetch_long_short_ratio`**: This is not a standard CCXT unified method. You *must* provide the specific exchange method name and its parameters within the `params` argument (e.g., `params={'method_name': 'fapiPublicGetGlobalLongShortAccountRatio', 'method_params': {'symbol': 'BTCUSDT', 'period': '5m'}}` for Binance futures).
*   **Error Handling**: Tools return a dictionary with an `"error"` key if an issue occurs during the CCXT call. 
