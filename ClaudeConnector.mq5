
//+------------------------------------------------------------------+
//|                                              ClaudeConnector.mq5 |
//|                                  Copyright 2025, Claude AI Team  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Claude AI Team"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Inputs
input string   ServerHost = "127.0.0.1";
input int      ServerPort = 5555;
input int      InpMagicNumber = 123456;
input double   MaxSlippage = 10;
input double   MaxSpread = 20;
input bool     BootstrapOnStart = true;     // Send a larger M1 history once on start
input int      BootstrapBars = 12000;       // ~8.3 days of M1 bars

//--- Globals
int socketHandle = INVALID_HANDLE;
CTrade trade;
datetime lastCandleTime = 0;
bool didBootstrap = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Initializing ClaudeConnector...");
   
   // Enable Timer for checking connection or health
   EventSetTimer(1);
   
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints((ulong)MaxSlippage);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // Note: We connect per-tick now for reliability
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(socketHandle != INVALID_HANDLE) {
      SocketClose(socketHandle);
   }
   EventKillTimer();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Send CLOSED M1 bars to Python. Python rebuilds 5m/15m/45m bars in UTC
   // using label='right', closed='left' (same as training) and only trades
   // on completed 5-minute bars.
   datetime currentCandleTime = iTime(_Symbol, PERIOD_M1, 0);
   if(currentCandleTime == lastCandleTime) return;
   
   lastCandleTime = currentCandleTime;
   
   // Open Connection (Fresh every time for reliability)
   if(!ConnectToServer()) return;
   
   // 1. Gather Data
   string jsonPayload = BuildJsonPayload();
   if(jsonPayload == "") {
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return;
   }
   
   // 2. Send Data
   if(!SendRequest(jsonPayload)) {
      Print("Failed to send request.");
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return;
   }
   
   // 3. Receive Response
   string response = ReceiveResponse();
   
   // Close immediately after receiving
   SocketClose(socketHandle);
   socketHandle = INVALID_HANDLE;

   if(response == "") {
      Print("Empty response. connection issues?");
      return;
   }
   
   // 4. Parse Actions and Trade
   ProcessResponse(response);
}

//+------------------------------------------------------------------+
//| Connect to Python Server                                         |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE) {
      Print("Error creating socket: ", GetLastError());
      return false;
   }
   
   if(!SocketConnect(socketHandle, ServerHost, ServerPort, 5000)) {
      Print("Error connecting to ", ServerHost, ":", ServerPort, " Error: ", GetLastError());
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return false;
   }
   
   Print("Connected to Python Server!");
   return true;
}

//+------------------------------------------------------------------+
//| Send JSON Request                                                |
//+------------------------------------------------------------------+
bool SendRequest(string json)
{
   // Format: [4 bytes length][JSON string]
   // MQL5 doesn't support easy struct packing for network like Python.
   // We can send length as uint.
   
   uchar data[];
   StringToCharArray(json, data, 0, WHOLE_ARRAY, CP_UTF8);
   int len = ArraySize(data);
   if(len > 0 && data[len-1] == 0) len--; // Remove null terminator only if present
   
   // Prepare header (Big Endian 4 bytes)
   uchar header[4];
   header[0] = (uchar)((len >> 24) & 0xFF);
   header[1] = (uchar)((len >> 16) & 0xFF);
   header[2] = (uchar)((len >> 8) & 0xFF);
   header[3] = (uchar)(len & 0xFF);
   
   // Send Header
   if(SocketSend(socketHandle, header, 4) != 4) return false;
   
   // Send Body
   if(SocketSend(socketHandle, data, len) != len) return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Receive JSON Response                                            |
//+------------------------------------------------------------------+
string ReceiveResponse()
{
   // Read Header (4 bytes)
   uchar header[4];
   uint lenRead = SocketRead(socketHandle, header, 4, 5000);
   if(lenRead != 4) return "";
   
   int msgLen = (header[0] << 24) + (header[1] << 16) + (header[2] << 8) + header[3];
   
   if(msgLen <= 0 || msgLen > 100000) {
      Print("Invalid message length received: ", msgLen);
      return "";
   }
   
   uchar data[];
   ArrayResize(data, msgLen);
   lenRead = SocketRead(socketHandle, data, msgLen, 5000);
   
   if(lenRead != msgLen) {
      Print("Incomplete read. Expected ", msgLen, " got ", lenRead);
      return "";
   }
   
   return CharArrayToString(data, 0, WHOLE_ARRAY, CP_UTF8);
}

//+------------------------------------------------------------------+
//| Build JSON Payload                                               |
//+------------------------------------------------------------------+
string BuildJsonPayload()
{
   string json = "{";

   // --- Time / Timezone Metadata ---
   // Training pipeline assumes UTC timestamps for session features.
   // MT5 bars are typically stamped in broker server time, so we send the
   // server↔UTC offset so Python can convert all bar times to UTC reliably.
   int utcOffsetSec = (int)(TimeCurrent() - TimeGMT()); // e.g. +7200 for UTC+2
   json += "\"time\":{";
   json += StringFormat("\"server\":%I64d,\"gmt\":%I64d,\"utc_offset_sec\":%d",
                        (long)TimeCurrent(), (long)TimeGMT(), utcOffsetSec);
   json += "},";
   
   // --- Rates ---
   json += "\"rates\":{";
   bool bootstrap = false;
   int barsToSend = 1;
   if(BootstrapOnStart && !didBootstrap) {
      bootstrap = true;
      barsToSend = BootstrapBars;
      didBootstrap = true;
   }
   json += "\"m1\":" + GetRatesJson(_Symbol, PERIOD_M1, barsToSend, 1);
   json += "},";
   
   json += StringFormat("\"bootstrap\":%d,", bootstrap ? 1 : 0);
   
   // --- Position ---
   json += "\"position\":{";
   double volume = 0;
   double price = 0;
   double sl = 0;
   double tp = 0;
   double profit = 0;
   datetime open_time = 0;
   int type = -1; // -1=None
   
   if(PositionSelect(_Symbol)) {
      volume = PositionGetDouble(POSITION_VOLUME);
      price = PositionGetDouble(POSITION_PRICE_OPEN);
      sl = PositionGetDouble(POSITION_SL);
      tp = PositionGetDouble(POSITION_TP);
      profit = PositionGetDouble(POSITION_PROFIT); // In currency
      open_time = (datetime)PositionGetInteger(POSITION_TIME);
      long posType = PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY) type = 0;
      else if(posType == POSITION_TYPE_SELL) type = 1;
   }
   
   json += StringFormat("\"type\":%d,\"volume\":%.2f,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"profit\":%.2f,\"open_time\":%I64d",
                        type, volume, price, sl, tp, profit, (long)open_time);
   json += "},";
   
   // --- Account ---
   json += "\"account\":{";
   json += StringFormat("\"balance\":%.2f,\"equity\":%.2f", 
                        AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));
   json += "}";
   
   json += "}";
   return json;
}

//+------------------------------------------------------------------+
//| Helper: Get Rates as JSON Array                                  |
//+------------------------------------------------------------------+
string GetRatesJson(string symbol, ENUM_TIMEFRAMES period, int count, int startPos)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true); // index 0 is most recent in the requested window
   int copied = CopyRates(symbol, period, startPos, count, rates);
   if(copied <= 0) return "[]";
   
   string json = "[";
   // Output oldest → newest
   for(int i=copied-1; i>=0; i--) {
      // [time, open, high, low, close]
      json += StringFormat("[%I64d,%.5f,%.5f,%.5f,%.5f]",
                           (long)rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close);
      if(i > 0) json += ",";
   }
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| Process Actions from Python                                      |
//+------------------------------------------------------------------+
void ProcessResponse(string json)
{
   // Crude JSON parsing
   // Expected: {"action": int, "size": double}
   
   int actionVal = (int)GetJsonValue(json, "action");
   double sizeVal = GetJsonValue(json, "size");
   double slVal = GetJsonValue(json, "sl");
   double tpVal = GetJsonValue(json, "tp");
   
   // Action: 0=Flat, 1=Long, 2=Short
   // 999 = No-op (keep state, no trade operation)
   if(actionVal == -1) return; // Parse error
   if(actionVal == 999) return; // No-op
   
   // Current Position
   bool hasPosition = PositionSelect(_Symbol);
   long currentType = -1;
   if(hasPosition) currentType = PositionGetInteger(POSITION_TYPE);
   
   // Execute based on Action
   // 0: Flat (Close if any)
   if(actionVal == 0) {
      if(hasPosition) {
         Print("AI says FLAT. Closing position.");
         trade.PositionClose(_Symbol);
      }
   }
   // 1: Long
   else if(actionVal == 1) {
      if(hasPosition && currentType == POSITION_TYPE_SELL) {
         Print("AI says LONG. Closing Short first.");
         trade.PositionClose(_Symbol);
         Sleep(500);
         hasPosition = false;
      }
      
      if(!hasPosition) {
         Print("AI says LONG. Buying ", sizeVal, " lots.");
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         // Note: Python sends specialized "size" (0.25-1.0).
         // We must map this to actual lots.
         // For now, let's treat "size" from Python as Lot Size directly?
         // bridge.py logic: size_pct * risk_multiplier.
         // If risk_multiplier=0.1 (lots), then size_val is e.g. 0.1 * 1.0 = 0.1 lots.
         // So we use it directly.
         
         double lots = NormalizeDouble(sizeVal, 2);
         if(lots > 0) {
            trade.Buy(lots, _Symbol, ask, slVal, tpVal, "AI Long");
         }
      }
   }
   // 2: Short
   else if(actionVal == 2) {
      if(hasPosition && currentType == POSITION_TYPE_BUY) {
         Print("AI says SHORT. Closing Long first.");
         trade.PositionClose(_Symbol);
         Sleep(500);
         hasPosition = false;
      }
      
      if(!hasPosition) {
         Print("AI says SHORT. Selling ", sizeVal, " lots.");
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double lots = NormalizeDouble(sizeVal, 2);
         if(lots > 0) {
            trade.Sell(lots, _Symbol, bid, slVal, tpVal, "AI Short");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Simple Regex-like JSON Value Extractor                           |
//+------------------------------------------------------------------+
double GetJsonValue(string json, string key)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start < 0) return -1;
   
   start += StringLen(search);
   int end = StringFind(json, ",", start);
   int end2 = StringFind(json, "}", start);
   
   if(end < 0) end = end2;
   if(end2 >= 0 && end2 < end) end = end2;
   
   if(end < 0) return -1;
   
   string valStr = StringSubstr(json, start, end - start);
   return StringToDouble(valStr);
}
