# 📊 Samsung Electronics Stock Trading with Deep Q-Network (DQN)

## 🚀 Introduction  

Stock market prediction and trading is a crucial area in financial analysis. Stock prices are influenced by various factors, such as market trends, economic indicators, and investor sentiment. This project focuses on analyzing and developing an intelligent trading system for **Samsung Electronics (005930.KS)** using Deep Q-Network (DQN) reinforcement learning.  

Using **Yahoo Finance data**, we apply **Exploratory Data Analysis (EDA), Feature Engineering, Preprocessing, and Deep Q-Network (DQN) reinforcement learning** to create an autonomous trading agent.  




---

## 🎯 Objectives  

✅ Retrieve Samsung Electronics stock data using `yfinance` 📈  
✅ Perform **EDA** to visualize trends & correlations 📊  
✅ Extract features like **RSI, MACD, Bollinger Bands, Moving Averages**  
✅ Preprocess the data for reinforcement learning (normalization, handling missing values)  
✅ Implement a **Deep Q-Network (DQN) agent** for autonomous stock trading �  
✅ Train the agent to make optimal BUY/SELL/HOLD decisions  
✅ Evaluate trading performance using **portfolio returns and Sharpe ratio**  
✅ Generate trading signals for future market conditions 📈 


---

## 🏗️ Project Workflow  

🔹 **Step 1:** Data Collection (Samsung Electronics from Yahoo Finance API)  
🔹 **Step 2:** Exploratory Data Analysis (EDA)  
🔹 **Step 3:** Feature Engineering (Technical Indicators: RSI, MACD, SMA)  
🔹 **Step 4:** Data Preprocessing (Normalization, Environment Setup)  
🔹 **Step 5:** DQN Agent Training (Reinforcement Learning)  
🔹 **Step 6:** Trading Performance Evaluation (Portfolio Returns)  
🔹 **Step 7:** Future Trading Signal Generation & Visualization  

---

## ⚙️ Installation & Setup  

### 📌 **1. Clone the Repository**  
```bash
git clone https://github.com/gamzeakkurt/deep-learning-stock-prediction.git
cd deep-learning-stock-prediction


```

### 📌 **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 📌 **3. Run the Python**  
```bash
python main.py
```
Open `main.py` and execute the code.

To run the full analysis and forecasting pipeline, execute the main.py file. This will sequentially:

- Load stock data from Yahoo Finance
- Perform exploratory data analysis (EDA)
- Process data with feature engineering and normalization
- Train and evaluate a Linear Regression model
- Train and evaluate an LSTM model
- Forecast future stock prices
- Visualize results and save predictions
---

## 📦 Dependencies  

The project requires the following libraries:  

```txt
yfinance
quantstats
ta
PyPortfolioOpt
pandas==1.3.5  
numpy  
matplotlib  
seaborn  
scikit-learn  
torch
torchvision
plotly  
```

You can install them using:  
```bash
pip install yfinance quantstats ta PyPortfolioOpt pandas numpy matplotlib seaborn scikit-learn torch torchvision plotly
```

---

## 📊 Exploratory Data Analysis (EDA)  

✅ **Stock Price Trends:** Visualize historical stock prices over time  
✅ **Moving Averages & Indicators:** Compute SMA, EMA, RSI, and MACD  
✅ **Correlation Analysis:** Analyze relationships between different stocks  

🔍 **Example Visualization:**  

<p align="center">
  <img src="EDA-images/output_59_0.png" width="600">
</p>


---



## 🏗️ DQN Model Architecture

The **Deep Q-Network (DQN)** model used for stock trading follows the reinforcement learning architecture below:

### **Model Overview**
- **State Space:** 6-dimensional vector including:
  - Normalized portfolio balance
  - Normalized position value  
  - Normalized net worth
  - RSI indicator (0-1)
  - MACD signal
  - Price vs SMA ratio

- **Action Space:** 3 discrete actions
  - **0:** HOLD (maintain current position)
  - **1:** BUY (purchase stocks)
  - **2:** SELL (sell all holdings)

- **Neural Network Architecture:**
  - **Input Layer:** 6 neurons (state features)
  - **Hidden Layers:** 3 fully connected layers with 128 neurons each
  - **Output Layer:** 3 neurons (Q-values for each action)
  - **Activation:** ReLU for hidden layers
  - **Dropout:** 0.2 for regularization

### **Training Parameters**
- **Optimizer:** Adam optimizer with learning rate 0.001
- **Loss Function:** Mean Squared Error (MSE)
- **Experience Replay:** Buffer size of 10,000 transitions
- **Epsilon-Greedy:** Starting at 1.0, decaying to 0.01
- **Target Network:** Updated every 10 episodes
- **Discount Factor:** 0.99

### **Reward Function**
The agent receives rewards based on:
- Portfolio net worth changes
- Transaction costs (0.1% per trade)
- Penalty for excessive holding without trading

This architecture enables the agent to learn optimal trading strategies through trial and error, maximizing long-term portfolio returns.



---

## 📈 Results & Evaluation  

The DQN trading agent is evaluated using the following metrics:  

📌 **Portfolio Return (%)** - Total return on investment  
📌 **Sharpe Ratio** - Risk-adjusted return measure  
📌 **Maximum Drawdown** - Largest portfolio decline  
� **Numbelr of Trades** - Trading frequency  
📌 **Win Rate** - Percentage of profitable trades  

🔍 **Example Trading Performance:**  

The DQN agent learns to:
- Identify optimal entry and exit points
- Manage risk through position sizing
- Adapt to changing market conditions
- Maximize long-term portfolio growth

Training progress shows convergence of the Q-network and improvement in trading performance over episodes.


## 📄 Additional Information
All details regarding the code, including visualization, results, and interpretation, are available in the 'YahooFinanceStockMarketAnalysis-Report.pdf' document. You can download it for an in-depth understanding of the analysis and forecasting process.

---

## 📌 Future Improvements  

🔹 **Advanced DQN Variants:** Implement Double DQN, Dueling DQN, and Rainbow DQN for improved performance and stability.

🔹 **Multi-Asset Trading:** Extend the framework to trade multiple Korean stocks simultaneously with portfolio optimization.

🔹 **Alternative RL Algorithms:** Compare DQN with Actor-Critic methods (A3C, PPO) and other reinforcement learning approaches.

🔹 **Feature Enhancement:** Incorporate additional technical indicators, market sentiment data, and macroeconomic factors.

🔹 **Risk Management:** Implement advanced risk management strategies including stop-loss, position sizing, and volatility-based adjustments.

🔹 **Real-time Trading:** Deploy the trained agent for live trading with proper backtesting and paper trading validation.


---

## 📜 License  

This project is licensed under the **MIT License**.  

---

## 🔍 Keywords
#samsungelectronics #dqn #reinforcementlearning #stocktrading #deeplearning #finance #korea #yahoofinance #pytorch #eda 



## 📬 Contact  

For any questions or suggestions, feel free to reach out. 



