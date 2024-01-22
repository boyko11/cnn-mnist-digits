# Pipeline class to tie everything together

import torch.nn as nn
import torch.optim as optim

from cnn import CNN
from data_loader import MNISTDataLoader
from data_transformer import StandardScalerTransformer
from model_evaluator import ModelAccuracyEvaluator
from model_trainer import CNNModelTrainer


class Pipeline:
    def __init__(self, data_loader, data_transformer, model_trainer, model_evaluator):
        self.data_loader = data_loader
        self.data_transformer = data_transformer
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator

    def run(self):
        trainloader, testloader = self.data_loader.load_data()
        trainloader = self.data_transformer.transform_data(trainloader)
        testloader = self.data_transformer.transform_data(testloader)

        net = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        self.model_trainer.train_model(net, trainloader, criterion, optimizer, num_epochs=3)

        accuracy = self.model_evaluator.evaluate_model(net, testloader)
        return accuracy


if __name__ == '__main__':
    mnist_loader = MNISTDataLoader()
    transformer = StandardScalerTransformer()
    trainer = CNNModelTrainer()
    evaluator = ModelAccuracyEvaluator()

    pipeline = Pipeline(mnist_loader, transformer, trainer, evaluator)
    accuracy = pipeline.run()
    print(f'Pipeline accuracy: {accuracy:.2f}%')
