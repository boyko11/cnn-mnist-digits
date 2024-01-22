# Abstract base class for model trainers
from abc import ABC, abstractmethod


class AbstractModelTrainer(ABC):
    @abstractmethod
    def train_model(self, model, trainloader, criterion, optimizer, num_epochs):
        pass


# Concrete implementation for training CNN models
class CNNModelTrainer(AbstractModelTrainer):
    def train_model(self, model, trainloader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 250 == 249:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f}')
                    running_loss = 0.0
        print('Finished Training')

# Example usage
# if __name__ == '__main__':
#     net = Net()  # Assuming Net is defined as before
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#     num_epochs = 3
#
#     trainer = CNNModelTrainer()
#     trainer.train_model(net, trainloader, criterion, optimizer, num_epochs)
