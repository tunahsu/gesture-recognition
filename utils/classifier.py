import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder


class Network(nn.Module):
    def __init__(self, input_size=22, hidden_size=128, num_classes=4):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    

class Classifier:
    def __init__(self, model_path):
        self.encoder = LabelEncoder()
        self.encoder.fit(['d', 'y', 'f', 'v'])
        self.emoji = {'d': '☝️', 'y': '🤙', 'f': '👌', 'v': '✌️'}
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self):
        self.model = Network()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def __call__(self, input_keypoint):
        if not type(input_keypoint) == torch.Tensor:
            input_keypoint = torch.tensor(input_keypoint, dtype=torch.float32)
        out = self.model(input_keypoint)
        _, predict = torch.max(out, -1)
        label_predict = self.encoder.inverse_transform([predict])[0]
        emoji = self.emoji[label_predict]
        return emoji


if __name__ == '__main__':
    classifier = Classifier('../models/pose_classification.pt')
    input = torch.randn(24)
    prediction = classifier(input)
    print(prediction)
