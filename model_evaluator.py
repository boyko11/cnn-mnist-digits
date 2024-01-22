# Abstract base class for model evaluators
from abc import ABC, abstractmethod

import torch


class AbstractModelEvaluator(ABC):
    @abstractmethod
    def evaluate_model(self, model, testloader):
        pass


# Concrete implementation for evaluating CNN model accuracy
class ModelAccuracyEvaluator(AbstractModelEvaluator):
    def evaluate_model(self, model, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        print(f'Number of correct predictions: {correct} out of {total}')
        return accuracy

# Example usage
# if __name__ == '__main__':
#     evaluator = ModelAccuracyEvaluator()
#     accuracy = evaluator.evaluate_model(net, testloader)
