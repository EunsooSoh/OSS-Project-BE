import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class StockTradingEnvironment:
    """
    Stock trading environment for DQN agent
    """
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(6)
        
        # Current price and technical indicators
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Portfolio state
        portfolio_state = [
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * current_price / self.initial_balance,  # Normalized position value
            self.net_worth / self.initial_balance,  # Normalized net worth
        ]
        
        # Market state (technical indicators)
        market_state = [
            self.data.iloc[self.current_step]['RSI'] / 100,  # Normalized RSI
            self.data.iloc[self.current_step]['MACD'],  # MACD
            (current_price - self.data.iloc[self.current_step]['SMA_20']) / current_price,  # Price vs SMA
        ]
        
        return np.array(portfolio_state + market_state, dtype=np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades.append(('SELL', self.current_step, current_price, self.shares_held))
                self.shares_held = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new net worth
        if self.current_step < len(self.data):
            new_price = self.data.iloc[self.current_step]['Close']
            self.net_worth = self.balance + self.shares_held * new_price
        
        # Calculate reward based on net worth change
        reward = (self.net_worth - self.max_net_worth) / self.initial_balance
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Add penalty for holding too long without trading
        if len(self.trades) == 0 and self.current_step > 50:
            reward -= 0.001
        
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {'net_worth': self.net_worth}

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for stock trading
    """
    def __init__(self, state_size=6, action_size=3, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    """
    DQN Agent for stock trading
    """
    def __init__(self, state_size=6, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_model(data, episodes=500, company='Samsung Electronics'):
    """
    Train DQN model for stock trading
    """
    print(f"\n========= Training DQN Model for {company} =========")
    
    # Prepare environment
    env = StockTradingEnvironment(data)
    agent = DQNAgent()
    
    episode_rewards = []
    episode_net_worths = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train the agent
        agent.replay()
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        episode_net_worths.append(env.net_worth)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Net Worth: ${env.net_worth:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    print(f"\nDQN Training completed for {company}")
    print(f"Final Net Worth: ${env.net_worth:.2f}")
    print(f"Total Return: {((env.net_worth - env.initial_balance) / env.initial_balance) * 100:.2f}%")
    
    return agent, env, episode_rewards, episode_net_worths

def test_dqn_model(agent, data, company='Samsung Electronics'):
    """
    Test the trained DQN model
    """
    print(f"\n========= Testing DQN Model for {company} =========")
    
    # Set epsilon to 0 for testing (no exploration)
    agent.epsilon = 0
    
    env = StockTradingEnvironment(data)
    state = env.reset()
    
    actions_taken = []
    portfolio_values = []
    
    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        actions_taken.append(action)
        portfolio_values.append(info['net_worth'])
        
        state = next_state
        
        if done:
            break
    
    print(f"Test completed for {company}")
    print(f"Final Net Worth: ${env.net_worth:.2f}")
    print(f"Total Return: {((env.net_worth - env.initial_balance) / env.initial_balance) * 100:.2f}%")
    print(f"Number of Trades: {len(env.trades)}")
    
    return {
        'net_worth': env.net_worth,
        'portfolio_values': portfolio_values,
        'actions': actions_taken,
        'trades': env.trades,
        'return_pct': ((env.net_worth - env.initial_balance) / env.initial_balance) * 100
    }

def forecast_dqn(agent, data, future_days=30):
    """
    Generate trading signals for future days using DQN
    """
    print(f"\n========= DQN Forecasting for next {future_days} days =========")
    
    # Use the last part of data to simulate future trading
    last_data = data.tail(100).copy()  # Use last 100 days as context
    
    # Generate future dates
    last_date = pd.to_datetime(data.index[-1]) if hasattr(data.index[-1], 'date') else pd.to_datetime(data['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')
    
    # Create mock future data (using last known values with small random variations)
    future_data = []
    last_price = data['Close'].iloc[-1]
    
    for i, date in enumerate(future_dates):
        # Simple random walk for price simulation
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = last_price * (1 + price_change)
        
        future_data.append({
            'Date': date,
            'Close': new_price,
            'RSI': 50 + np.random.normal(0, 10),  # Random RSI around 50
            'MACD': np.random.normal(0, 0.1),
            'SMA_20': new_price * 0.98  # Slightly below current price
        })
        
        last_price = new_price
    
    future_df = pd.DataFrame(future_data)
    
    # Set epsilon to 0 for prediction (no exploration)
    agent.epsilon = 0
    
    env = StockTradingEnvironment(future_df)
    state = env.reset()
    
    predictions = []
    
    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        predictions.append({
            'Date': future_df.iloc[env.current_step-1]['Date'],
            'Predicted_Price': future_df.iloc[env.current_step-1]['Close'],
            'Action': action_name,
            'Portfolio_Value': info['net_worth']
        })
        
        state = next_state
        
        if done:
            break
    
    forecast_df = pd.DataFrame(predictions)
    forecast_df.set_index('Date', inplace=True)
    
    print("DQN Forecast completed")
    print(f"Recommended actions for next {future_days} days:")
    print(forecast_df[['Action', 'Predicted_Price']].head(10))
    
    return forecast_df
def 
plot_dqn_training_progress(episode_rewards, episode_net_worths):
    """
    Plot DQN training progress
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('DQN Training Progress - Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot net worth progression
    ax2.plot(episode_net_worths)
    ax2.set_title('DQN Training Progress - Portfolio Net Worth')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Net Worth ($)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('Results/dqn_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_dqn_portfolio_performance(test_results):
    """
    Plot DQN portfolio performance during testing
    """
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio value over time
    plt.subplot(2, 1, 1)
    plt.plot(test_results['portfolio_values'])
    plt.title('DQN Portfolio Performance - Net Worth Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot actions taken
    plt.subplot(2, 1, 2)
    actions = test_results['actions']
    action_names = ['HOLD', 'BUY', 'SELL']
    action_colors = ['gray', 'green', 'red']
    
    for i, action in enumerate(actions):
        plt.scatter(i, action, c=action_colors[action], alpha=0.6, s=20)
    
    plt.title('DQN Trading Actions Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Action (0=HOLD, 1=BUY, 2=SELL)')
    plt.yticks([0, 1, 2], action_names)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Results/dqn_portfolio_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPortfolio Performance Summary:")
    print(f"Final Return: {test_results['return_pct']:.2f}%")
    print(f"Number of Trades: {len(test_results['trades'])}")

def plot_dqn_forecast_results(historical_data, forecast_data):
    """
    Plot DQN forecast results
    """
    plt.figure(figsize=(15, 10))
    
    # Plot historical prices
    plt.subplot(2, 1, 1)
    historical_dates = pd.to_datetime(historical_data['Date']) if 'Date' in historical_data.columns else historical_data.index
    plt.plot(historical_dates[-100:], historical_data['Close'].tail(100), label='Historical Prices', color='blue')
    plt.plot(forecast_data.index, forecast_data['Predicted_Price'], label='Predicted Prices', color='red', linestyle='--')
    plt.title('Samsung Electronics Stock Price - Historical vs DQN Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (KRW)')
    plt.legend()
    plt.grid(True)
    
    # Plot trading actions
    plt.subplot(2, 1, 2)
    buy_signals = forecast_data[forecast_data['Action'] == 'BUY']
    sell_signals = forecast_data[forecast_data['Action'] == 'SELL']
    hold_signals = forecast_data[forecast_data['Action'] == 'HOLD']
    
    plt.plot(forecast_data.index, forecast_data['Predicted_Price'], color='black', alpha=0.7)
    
    if not buy_signals.empty:
        plt.scatter(buy_signals.index, buy_signals['Predicted_Price'], 
                   color='green', marker='^', s=100, label='BUY Signal', alpha=0.8)
    
    if not sell_signals.empty:
        plt.scatter(sell_signals.index, sell_signals['Predicted_Price'], 
                   color='red', marker='v', s=100, label='SELL Signal', alpha=0.8)
    
    plt.title('DQN Trading Signals for Samsung Electronics')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (KRW)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Results/dqn_forecast_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print action summary
    action_counts = forecast_data['Action'].value_counts()
    print(f"\nDQN Trading Signal Summary:")
    for action, count in action_counts.items():
        print(f"{action}: {count} days ({count/len(forecast_data)*100:.1f}%)")