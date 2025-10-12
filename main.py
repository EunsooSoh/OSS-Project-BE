# Import libraries
from data_acquisition import *
from FeatureEngineering import *
from Preprocessing import *
from linear_regression import *
from eda import perform_eda
from dqn_model import *

# Define ticker for Samsung Electronics
tickers = ['005930.KS']  # Samsung Electronics Co., Ltd.
companies = ['Samsung Electronics']

# Load data
df = load_stock_data(tickers)

# Perform Exploratory Data Analysis (EDA)
perform_eda(df, tickers, companies)

# Feature Engineering for Samsung Electronics Stock
df_samsung = download_and_process_data('005930.KS')

# Preprocessing(Normalizing)
scaler, normalized_data = min_max_scaling(df_samsung)
df_normalized = pd.DataFrame(normalized_data, columns=df_samsung.columns)
df_normalized["Date"] = df_samsung.index
df_normalized = df_normalized.set_index('Date')

# PyTorch Linear Regression Model
linear_model, scaler, mae, mse, rmse, r2, compare_df = linear_prediction(df_normalized)

# Future Forecasting with PyTorch Linear Regression
linear_model, future_predictions = linear_forecasting(df_samsung, linear_model, scaler, future_days=30)

# Saving Future Prediction Results
future_predictions.to_csv('Results/future_predictions-samsung-pytorch-linear-regression.csv')

# Filling Nan Values
df_normalized = df_normalized.fillna(0)

print("\n" + "="*60)
print("TRAINING DQN MODEL FOR SAMSUNG ELECTRONICS STOCK TRADING")
print("="*60)

# Train DQN Model
dqn_agent, dqn_env, episode_rewards, episode_net_worths = train_dqn_model(df_samsung, episodes=500, company='Samsung Electronics')

# Test DQN Model
test_results = test_dqn_model(dqn_agent, df_samsung, company='Samsung Electronics')

# Plot DQN Training Progress
plot_dqn_training_progress(episode_rewards, episode_net_worths)

# Plot DQN Portfolio Performance
plot_dqn_portfolio_performance(test_results)

# DQN Forecasting
dqn_forecast = forecast_dqn(dqn_agent, df_samsung, future_days=30)
dqn_forecast.to_csv('Results/future_predictions_samsung_dqn.csv')

# Plot DQN Forecast Results
plot_dqn_forecast_results(df_samsung, dqn_forecast)

print("\n" + "="*60)
print("SAMSUNG ELECTRONICS DQN MODEL TRAINING AND TESTING COMPLETED")
print("="*60)

