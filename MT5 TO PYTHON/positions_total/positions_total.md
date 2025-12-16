**positions_total**  
Get the number of open positions.  

| positions_total() |
| ----------------- |
  
**Return Value**  
Integer value.  
**Note**  
The function is similar to ++[PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal)++.  
The function is similar to ++[PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal)++.  
**Example:**  

| import MetaTrader5 as mt5 # display data on the MetaTrader 5 package print("MetaTrader5 package author: ",mt5.__author__) print("MetaTrader5 package version: ",mt5.__version__)   # establish connection to MetaTrader 5 terminal if not mt5.initialize():     print("initialize() failed, error code =",mt5.last_error())     quit()   # check the presence of open positions positions_total=mt5.positions_total() if positions_total>0:     print("Total positions=",positions_total) else:     print("Positions not found")   # shut down connection to the MetaTrader 5 terminal mt5.shutdown() |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  
