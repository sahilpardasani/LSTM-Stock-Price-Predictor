# training_prediction.py

import torch
from model import StockLSTM  
import numpy as np

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save scaler bounds (to be updated in load_data)
error_feature_min = 0.0
error_feature_max = 1.0

def set_error_scaling_bounds(min_val, max_val): #used during data loading to get min/max values of error column and inject it into global variable
    global error_feature_min, error_feature_max
    error_feature_min = min_val
    error_feature_max = max_val

def train_model(X, y, num_epochs=50, lr=0.001):
    model = StockLSTM(input_size=6).to(device)  # initialise the model with 6 input features
    criterion = torch.nn.MSELoss() #loss MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # adam optimizer

    X_train = torch.Tensor(X).to(device) #convert numpy arrays to tensors
    y_train = torch.Tensor(y).unsqueeze(1).to(device) #change shape from [n] to [n,1] to match pred shape

    for epoch in range(num_epochs): #trains the model to minimise dfference between predicted and actual closing price 
        model.train()
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def predict_next(model, last_90_days, scaler, last_error=None): #predict the next day's closing price 
    model.eval()
    input_seq = last_90_days.copy() #creates a copy of the input

    if last_error is not None:
        # Apply correct scaling for the error column using global min/max
        scaled_error = (last_error - error_feature_min) / (error_feature_max - error_feature_min)
        scaled_error = np.clip(scaled_error, 0.0, 1.0)  
        input_seq[-1][-1] = scaled_error  # inject scaled error into last timestep

    input_tensor = torch.Tensor(input_seq).unsqueeze(0).to(device) #adds a batch dimension shape [1, 90, 6]

    with torch.no_grad(): #disable gradient tracking for efficency 
        prediction = model(input_tensor).cpu().item() #grabs the predicted scaled price

    dummy = np.zeros((1, 6))
    dummy[0][3] = prediction  # only scale Close
    predicted_price = scaler.inverse_transform(dummy)[0][3]
#uses trained minmax scaler to convert the scaled price back to original price
    return round(predicted_price, 2)
