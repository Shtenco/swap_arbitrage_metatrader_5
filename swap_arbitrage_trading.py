
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import scipy.optimize as sco
import random

class SwapArbitrageAnalyzer:
    def __init__(self, config=None):
        self.config = {
            'target_volume': 100.0,
            'max_pairs': 28,
            'leverage': 1,  # Leverage 1:10
            'broker_suffix': '',
            'risk_free_rate': 0.001,
            'optimization_period': int((datetime(2025, 3, 17) - datetime(2015, 1, 1)).days),  # From 01.01.2015 to 17.03.2025
            'panel_width': 750,
            'panel_height': 500,
            'risk_aversion': 2.0,
            'swap_weight': 0.3,
            'return_weight': 0.6,
            'volatility_weight': 0.1,
            'simulation_days': int((datetime(2025, 3, 17) - datetime(2015, 1, 1)).days),
            'monthly_deposit_rate': 0.02  # 2% of initial capital monthly
        }
        if config:
            self.config.update(config)
            
        self.currencies = ["EUR", "GBP", "USD", "JPY", "CHF", "AUD", "NZD", "CAD"]
        self.pairs = []
        self.market_rates = {}
        self.swap_info = {}
        self.initialized = False
        self.optimal_portfolio = None
        self.pair_performance = {}
        self.portfolio_history = {'dates': [], 'equity': [], 'swap': [], 'equity_with_swap': [], 'equity_with_reinvestment': []}
        self._generate_all_pairs()
        
    def _generate_all_pairs(self):
        self.pairs = []
        for i, base in enumerate(self.currencies):
            for quote in self.currencies[i+1:]:
                if self._get_currency_priority(base) < self._get_currency_priority(quote):
                    self.pairs.append(f"{base}{quote}{self.config['broker_suffix']}")
                else:
                    self.pairs.append(f"{quote}{base}{self.config['broker_suffix']}")
    
    def _get_currency_priority(self, currency):
        priorities = {"EUR": 1, "GBP": 2, "AUD": 3, "NZD": 4, "USD": 5, "CAD": 6, "CHF": 7, "JPY": 8}
        return priorities.get(currency, 9)
        
    def initialize(self):
        if not mt5.initialize():
            print(f"MetaTrader5 initialization failed, error={mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if not account_info:
            print("Failed to get account information")
            return False
            
        print(f"MetaTrader5 initialized. Account: {account_info.login}, Balance: {account_info.balance}")
        self._get_current_market_rates()
        self._init_swap_data()
        self.initialized = True
        return True
    
    def analyze(self):
        if not self.initialized and not self.initialize():
            return False
        self.optimal_portfolio = self._optimize_portfolio()
        if self.optimal_portfolio:
            self._simulate_portfolio_performance()
            self._create_visualizations()
        return True
    
    def _get_current_market_rates(self):
        for pair in self.pairs:
            symbol = pair
            if not mt5.symbol_select(symbol, True):
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                reversed_symbol = f"{quote_currency}{base_currency}{self.config['broker_suffix']}"
                if mt5.symbol_select(reversed_symbol, True):
                    is_direct = False
                    symbol = reversed_symbol
                else:
                    self.market_rates[pair] = {'bid': 0, 'ask': 0, 'is_direct': True}
                    continue
            else:
                is_direct = True
            
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                bid, ask = tick.bid, tick.ask
                if not is_direct and ask != 0:
                    bid, ask = 1.0 / ask, 1.0 / bid
                self.market_rates[pair] = {'bid': bid, 'ask': ask, 'is_direct': is_direct, 'symbol': symbol}
    
    def _init_swap_data(self):
        print("Initializing swap and return data from 01.01.2015...")
        available_pairs = 0
        start_date = datetime(2015, 1, 1)
        for pair in self.pairs:
            symbol_info = mt5.symbol_info(pair)
            if not symbol_info:
                continue
                
            swap_long = symbol_info.swap_long
            swap_short = symbol_info.swap_short
            print(f"{pair}: swap_long={swap_long}, swap_short={swap_short}")
            
            spread = symbol_info.spread * symbol_info.point
            swap_ratio = max(abs(swap_long), abs(swap_short)) / spread if spread > 0 else 0
            
            history = self._get_historical_data(pair, start_date)
            if history is not None:
                available_pairs += 1
                direction = 'long' if swap_long > swap_short else 'short'
                self.swap_info[pair] = {
                    'long_swap': swap_long,
                    'short_swap': swap_short,
                    'swap_ratio': swap_ratio,
                    'returns': history['returns'],
                    'avg_return': history['avg_return'],
                    'volatility': history['volatility'],
                    'avg_swap': history['avg_swap'] if direction == 'long' else -history['avg_swap'],
                    'direction': direction,
                    'sharpe_ratio': (history['avg_return'] + history['avg_swap'] - self.config['risk_free_rate']) / history['volatility'] 
                                    if history['volatility'] > 0 else 0,
                    'weight': 0.0,
                    'data': history['data']
                }
        print(f"Available pairs with data: {available_pairs}")
    
    def _get_historical_data(self, symbol, start_date):
        try:
            now = datetime.now()
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date, now)
            if rates is None or len(rates) < 10:
                print(f"Insufficient data for {symbol}: {len(rates) if rates is not None else 'None'} bars")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df['return'] = df['close'].pct_change()
            
            symbol_info = mt5.symbol_info(symbol)
            best_swap = max(symbol_info.swap_long, symbol_info.swap_short)
            swap_in_points = best_swap if symbol_info.swap_long > symbol_info.swap_short else -best_swap
            point_value = symbol_info.point
            df['swap_return'] = (swap_in_points * point_value) / df['close'] * self.config['leverage']  # Account for leverage
            
            self.pair_performance[symbol] = {
                'dates': df.index.tolist(),
                'prices': df['close'].tolist(),
                'returns': df['return'].dropna().tolist(),
                'swap_returns': df['swap_return'].tolist()
            }
            
            print(f"{symbol}: collected {len(df)} bars, returns={len(df['return'].dropna())}")
            return {
                'returns': df['return'].dropna().values,
                'avg_return': df['return'].mean(),
                'volatility': df['return'].std(),
                'avg_swap': df['swap_return'].mean(),
                'data': df
            }
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _optimize_portfolio(self):
        print("Optimizing portfolio from 01.01.2015 with positive swap...")
        eligible_pairs = [pair for pair, info in self.swap_info.items() if len(info['returns']) >= 10]
        
        if len(eligible_pairs) < 1:
            print(f"Insufficient data for portfolio optimization. Found pairs: {len(eligible_pairs)}")
            return None
        
        num_pairs = random.randint(1, min(self.config['max_pairs'], len(eligible_pairs)))
        eligible_pairs = random.sample(eligible_pairs, num_pairs)
        print(f"Selected pairs for optimization: {len(eligible_pairs)}")
        
        # First ensure all return arrays are of the same length
        min_length = min(len(self.swap_info[pair]['returns']) for pair in eligible_pairs)
        
        returns_data = {pair: (self.swap_info[pair]['returns'][:min_length] * self.config['leverage'] + self.swap_info[pair]['avg_swap']) 
                       if self.swap_info[pair]['direction'] == 'long' 
                       else (-self.swap_info[pair]['returns'][:min_length] * self.config['leverage'] + self.swap_info[pair]['avg_swap']) 
                       for pair in eligible_pairs}
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov()
        
        expected_returns = {}
        for pair in eligible_pairs:
            market_return = self.swap_info[pair]['avg_return'] * self.config['leverage'] if self.swap_info[pair]['direction'] == 'long' else -self.swap_info[pair]['avg_return'] * self.config['leverage']
            swap_return = self.swap_info[pair]['avg_swap']
            volatility = self.swap_info[pair]['volatility'] * self.config['leverage']
            
            norm_market = (market_return - min([self.swap_info[p]['avg_return'] * self.config['leverage'] if self.swap_info[p]['direction'] == 'long' 
                                               else -self.swap_info[p]['avg_return'] * self.config['leverage'] for p in eligible_pairs])) / \
                         (max([self.swap_info[p]['avg_return'] * self.config['leverage'] if self.swap_info[p]['direction'] == 'long' 
                               else -self.swap_info[p]['avg_return'] * self.config['leverage'] for p in eligible_pairs]) - 
                          min([self.swap_info[p]['avg_return'] * self.config['leverage'] if self.swap_info[p]['direction'] == 'long' 
                               else -self.swap_info[p]['avg_return'] * self.config['leverage'] for p in eligible_pairs]) + 1e-10)
            norm_swap = (swap_return - min([self.swap_info[p]['avg_swap'] for p in eligible_pairs])) / \
                       (max([self.swap_info[p]['avg_swap'] for p in eligible_pairs]) - 
                        min([self.swap_info[p]['avg_swap'] for p in eligible_pairs]) + 1e-10)
            norm_vol = (volatility - min([self.swap_info[p]['volatility'] * self.config['leverage'] for p in eligible_pairs])) / \
                      (max([self.swap_info[p]['volatility'] * self.config['leverage'] for p in eligible_pairs]) - 
                       min([self.swap_info[p]['volatility'] * self.config['leverage'] for p in eligible_pairs]) + 1e-10)
            
            combined_score = (self.config['swap_weight'] * norm_swap + 
                             self.config['return_weight'] * norm_market - 
                             self.config['volatility_weight'] * norm_vol)
            expected_returns[pair] = combined_score
        
        def portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate):
            weights = np.array(weights)
            returns = np.sum(expected_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, std, (returns - risk_free_rate) / std if std > 0 else 0
        
        def objective(weights, expected_returns, cov_matrix, risk_free_rate):
            returns, std, sharpe = portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate)
            return -sharpe
        
        # Constraint for positive swap
        def swap_constraint(weights, eligible_pairs):
            total_swap = np.sum([self.swap_info[pair]['avg_swap'] * abs(weights[i]) for i, pair in enumerate(eligible_pairs)])
            return total_swap  # Must be >= 0
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1},  # Sum of weights = 1
            {'type': 'ineq', 'fun': lambda x: swap_constraint(x, eligible_pairs)}  # Swap >= 0
        ]
        bounds = tuple((-1, 1) for _ in range(len(eligible_pairs)))
        initial_weights = np.array([1.0 / len(eligible_pairs) if self.swap_info[pair]['direction'] == 'long' 
                                   else -1.0 / len(eligible_pairs) for pair in eligible_pairs])
        
        result = sco.minimize(
            objective,
            initial_weights,
            args=(np.array(list(expected_returns.values())), cov_matrix.values, self.config['risk_free_rate']),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result['x'] if result['success'] else initial_weights
        mask = np.abs(optimal_weights) < 0.05
        optimal_weights[mask] = 0
        if np.sum(np.abs(optimal_weights)) == 0:
            optimal_weights = initial_weights
        else:
            optimal_weights = optimal_weights / np.sum(np.abs(optimal_weights))
        
        optimal_portfolio = {}
        for i, pair in enumerate(eligible_pairs):
            if optimal_weights[i] != 0:
                optimal_portfolio[pair] = optimal_weights[i]
                self.swap_info[pair]['weight'] = optimal_weights[i] * 100
        
        print("\nOptimal portfolio with positive swap:")
        for pair, weight in sorted(optimal_portfolio.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = 'Long' if weight > 0 else 'Short'
            swap_value = self.swap_info[pair]['long_swap'] if weight > 0 else self.swap_info[pair]['short_swap']
            print(f"Pair: {pair}, Direction: {direction}, Weight: {abs(weight)*100:.2f}%, Swap: {swap_value:.2f}")
        
        portfolio_return = np.sum([(self.swap_info[pair]['avg_return'] * self.config['leverage'] + self.swap_info[pair]['avg_swap']) * optimal_portfolio.get(pair, 0) 
                                  if optimal_portfolio.get(pair, 0) > 0 
                                  else (-self.swap_info[pair]['avg_return'] * self.config['leverage'] + self.swap_info[pair]['avg_swap']) * optimal_portfolio.get(pair, 0) 
                                  for pair in eligible_pairs])
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix.values, optimal_weights)))
        portfolio_swap = np.sum([self.swap_info[pair]['avg_swap'] * abs(optimal_portfolio.get(pair, 0)) 
                                for pair in eligible_pairs])
        
        return {
            'weights': optimal_portfolio,
            'return': portfolio_return,
            'risk': portfolio_std,
            'swap': portfolio_swap,
            'sharpe': (portfolio_return - self.config['risk_free_rate']) / portfolio_std if portfolio_std > 0 else 0
        }
    
    def _simulate_portfolio_performance(self):
        if not self.optimal_portfolio:
            print("Optimal portfolio not created")
            return
        
        initial_capital = 10000
        current_capital = initial_capital
        current_capital_with_swap = initial_capital
        current_capital_with_reinvestment = initial_capital
        dates = []
        equity = []
        swap_profit = []
        equity_with_swap = []
        equity_with_reinvestment = []
        
        all_dates = []
        for pair in self.optimal_portfolio['weights'].keys():
            all_dates.extend(self.swap_info[pair]['data'].index)
        all_dates = sorted(set(all_dates))
        
        start_date = datetime(2015, 1, 1)
        all_dates = [d for d in all_dates if d >= start_date]
        last_deposit_date = start_date
        
        monthly_deposit = initial_capital * self.config['monthly_deposit_rate']
        
        for date in all_dates:
            daily_return = 0
            daily_swap = 0
            for pair, weight in self.optimal_portfolio['weights'].items():
                if date in self.swap_info[pair]['data'].index:
                    pair_return = self.swap_info[pair]['data'].loc[date, 'return'] if not pd.isna(self.swap_info[pair]['data'].loc[date, 'return']) else 0
                    pair_swap = self.swap_info[pair]['data'].loc[date, 'swap_return']
                    if weight > 0:
                        daily_return += pair_return * weight * self.config['leverage']
                        daily_swap += pair_swap * abs(weight)
                    else:
                        daily_return += -pair_return * abs(weight) * self.config['leverage']
                        daily_swap += pair_swap * abs(weight)
            
            is_weekend = date.weekday() >= 5
            daily_swap_applied = 0 if is_weekend else daily_swap * initial_capital
            
            # Without swap (market returns only)
            market_profit = current_capital * daily_return
            current_capital += market_profit
            
            # With swap
            market_profit_with_swap = current_capital_with_swap * daily_return
            current_capital_with_swap += market_profit_with_swap + daily_swap_applied
            
            # With deposits and reinvestment
            if (date - last_deposit_date).days >= 30:  # Monthly deposit
                profit = current_capital_with_reinvestment - initial_capital
                current_capital_with_reinvestment += monthly_deposit
                if profit > 0:
                    current_capital_with_reinvestment += profit  # Reinvestment of profit
                initial_capital = current_capital_with_reinvestment  # Update initial capital for next month
                last_deposit_date = date
            
            market_profit_reinvest = current_capital_with_reinvestment * daily_return
            current_capital_with_reinvestment += market_profit_reinvest + daily_swap_applied
            
            dates.append(date)
            equity.append(current_capital)
            swap_profit.append(daily_swap_applied)
            equity_with_swap.append(current_capital_with_swap)
            equity_with_reinvestment.append(current_capital_with_reinvestment)
        
        self.portfolio_history = {
            'dates': dates,
            'equity': equity,
            'swap': swap_profit,
            'equity_with_swap': equity_with_swap,
            'equity_with_reinvestment': equity_with_reinvestment,
            'initial_capital': initial_capital
        }
        total_return = (equity[-1] - initial_capital) / initial_capital * 100
        total_return_with_swap = (equity_with_swap[-1] - initial_capital) / initial_capital * 100
        total_return_with_reinvestment = (equity_with_reinvestment[-1] - initial_capital) / initial_capital * 100
        print(f"Total return (without swap): {total_return:.2f}%")
        print(f"Total return (with swap): {total_return_with_swap:.2f}%")
        print(f"Total return (with deposits and reinvestment): {total_return_with_reinvestment:.2f}%")
    
    def _create_visualizations(self):
        if not self.optimal_portfolio or not self.portfolio_history:
            print("Data for visualization not ready")
            return
        
        # 1. Portfolio and its proportions
        plt.figure(figsize=(self.config['panel_width']/100, self.config['panel_height']/100), dpi=100)
        sorted_weights = sorted(self.optimal_portfolio['weights'].items(), key=lambda x: abs(x[1]), reverse=True)
        pairs = [f"{item[0]} ({'L' if item[1] > 0 else 'S'})" for item in sorted_weights]
        weights = [abs(item[1]) * 100 for item in sorted_weights]
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(pairs)))
        plt.pie(weights, labels=pairs, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 8})
        plt.title('Portfolio Proportions (L=Long, S=Short)')
        plt.tight_layout()
        plt.savefig('portfolio_proportions.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. Returns chart (without swap)
        plt.figure(figsize=(self.config['panel_width']/100, self.config['panel_height']/100), dpi=100)
        plt.plot(self.portfolio_history['dates'], self.portfolio_history['equity'], 'b-')
        plt.title('Portfolio Returns (without swap)')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_returns.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3. Accumulated swap chart
        plt.figure(figsize=(self.config['panel_width']/100, self.config['panel_height']/100), dpi=100)
        cumulative_swap = np.cumsum(self.portfolio_history['swap'])
        plt.plot(self.portfolio_history['dates'], cumulative_swap, 'g-')
        plt.title('Accumulated Swap')
        plt.ylabel('Swap ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_swap.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 4. Returns chart (with swap)
        plt.figure(figsize=(self.config['panel_width']/100, self.config['panel_height']/100), dpi=100)
        plt.plot(self.portfolio_history['dates'], self.portfolio_history['equity_with_swap'], 'r-')
        plt.title('Portfolio Returns (with swap)')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_with_swap.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 5. Returns chart (with deposits and reinvestment)
        plt.figure(figsize=(self.config['panel_width']/100, self.config['panel_height']/100), dpi=100)
        plt.plot(self.portfolio_history['dates'], self.portfolio_history['equity_with_reinvestment'], 'm-')
        plt.title('Portfolio Returns (with deposits and reinvestment)')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_with_reinvestment.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to separate files")

if __name__ == "__main__":
    analyzer = SwapArbitrageAnalyzer()
    if analyzer.initialize():
        analyzer.analyze()
