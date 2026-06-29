import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import flwr as fl
import numpy as np

class CartilageDegradationODE(nn.Module):
    """
    Neural ODE modeling cartilage degradation under Galactic Cosmic Radiation (GCR).
    State variables (y):
    - y[0]: Cartilage Matrix Integrity (0-1)
    - y[1]: 15-PGDH concentration
    - y[2]: Drug concentration (15-PGDH inhibitor)
    """
    def __init__(self):
        super(CartilageDegradationODE, self).__init__()
        # Neural network to approximate the dynamics
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        )

    def forward(self, t, y):
        """
        Calculates the derivatives dy/dt.
        """
        return self.net(y)


class SpacecraftClient(fl.client.NumPyClient):
    """
    Flower Client representing a simulated Spacecraft Node.
    Trains the Neural ODE on local synthetic biometric data.
    """
    def __init__(self, model, data, epochs=5):
        self.model = model
        self.data = data # Dictionary with 't' and 'y_true'
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        t = self.data['t']
        y_true = self.data['y_true']
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            # Initial state
            y0 = y_true[0]
            # Predict trajectory
            y_pred = odeint(self.model, y0, t)
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()
            
        return self.get_parameters(config={}), len(self.data['t']), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        t = self.data['t']
        y_true = self.data['y_true']
        
        with torch.no_grad():
            y0 = y_true[0]
            y_pred = odeint(self.model, y0, t)
            loss = self.criterion(y_pred, y_true)
            
        return float(loss), len(self.data['t']), {"mse": float(loss)}


def simulate_federated_learning(num_clients=3, rounds=3):
    """
    Simulates a federated learning network with multiple spacecrafts.
    """
    # 1. Create global model
    global_model = CartilageDegradationODE()

    # 2. Generate synthetic data for clients
    def generate_synthetic_data(num_points=100):
        t = torch.linspace(0., 10., num_points)
        # Synthetic y_true: [Matrix, 15-PGDH, Drug]
        y_true = torch.zeros(num_points, 3)
        y_true[:, 0] = torch.exp(-0.1 * t) + 0.05 * torch.randn(num_points) # Matrix degrades
        y_true[:, 1] = 1 - torch.exp(-0.2 * t) + 0.05 * torch.randn(num_points) # 15-PGDH increases
        y_true[:, 2] = 0.5 * torch.sin(t) + 0.5 # Oscillating drug concentration
        return {'t': t, 'y_true': y_true}

    client_data = [generate_synthetic_data() for _ in range(num_clients)]

    # 3. Define client function
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return SpacecraftClient(model=CartilageDegradationODE(), data=client_data[idx]).to_client()
        
    # 4. Define server strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # 5. Start simulation
    print("Starting Federated Learning Simulation for ChondroZero-G Digital Twin...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    return history, global_model

if __name__ == "__main__":
    # Test execution
    history, trained_model = simulate_federated_learning()
    print("Federated Learning Simulation Complete.")
