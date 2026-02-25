# Stock-Price-Prediction

## AIM
To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.

## Dataset

<img width="810" height="988" alt="image" src="https://github.com/user-attachments/assets/04987b96-1bcb-4af3-b541-2b48491c4837" />
<img width="820" height="970" alt="image" src="https://github.com/user-attachments/assets/038da5c9-5d51-4e2a-a554-f33f47ae367c" />

## Design Steps

### Step 1:
Import necessary libraries.

### Step 2:
Load and preprocess the data.

### Step 3:
Create input-output sequences.

### Step 2:
Convert data to PyTorch tensors.

### Step 3:
Define the RNN model.

### Step 2:
Train the model using the training data.

### Step 3:
Evaluate the model and plot predictions.

## Program
### Name: DHANUJA M
### Register Number: 212224230057


```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self,x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="853" height="645" alt="image" src="https://github.com/user-attachments/assets/3e2a7225-5168-42e9-9526-0b3a3566c7e3" />

### Predictions 

<img width="1329" height="797" alt="image" src="https://github.com/user-attachments/assets/6cecea05-8d86-4969-9158-18f5cbd76b23" />


## Result
The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.
