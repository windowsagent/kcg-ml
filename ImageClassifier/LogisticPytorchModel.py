import os
import numpy as np
import json
import torch
import argparse

class LogisticRegressionPytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim = 1, date_trained=None, tag=None):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.date_trained = date_trained
        self.tag = tag

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train_loop(self, train_emb, train_labels, epochs: int = 20000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_emb = torch.from_numpy(train_emb.astype(np.float32)).to(device)
        train_labels = torch.from_numpy(train_labels.astype(np.float32)).view(train_labels.shape[0], 1).to(device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        for epoch in range(epochs + 1):
            y_prediction = self(train_emb)
            loss = criterion(y_prediction, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self

    def save_model(self, file_path: str):
        torch.save(self.state_dict(), file_path)

    @staticmethod
    def load_model(file_path: str, input_dim: int, output_dim: int, date_trained=None, tag=None):
        model = LogisticRegressionPytorch(input_dim, output_dim, date_trained, tag)
        model.load_state_dict(torch.load(file_path))
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a logistic regression model.")
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file.')
    parser.add_argument('--model_file', type=str, default='torchlogisticregression_pixel-art.pt', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs.')
    args = parser.parse_args()

    # Load the JSON data
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # Extract clip vectors and set tag as 1 (assuming all data in JSON is tagged)
    clip_vectors = np.array([entry['clip_vector'] for entry in data])
    tags = np.array([1 for _ in range(len(clip_vectors))])  # Tag is set to 1 for all data points

    # Initialize the logistic regression model
    model = LogisticRegressionPytorch(clip_vectors.shape[1], 1, date_trained='2023-05-10', tag='pixel-art')

    # Train the model
    model.train_loop(clip_vectors, tags, epochs=args.epochs)

    # Save the model
    model.save_model(args.model_file)
