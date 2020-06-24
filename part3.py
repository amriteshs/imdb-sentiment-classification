import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.gru = tnn.GRU(input_size=50, hidden_size=75, batch_first=True, dropout=0.2, num_layers=4)
        self.fc1 = tnn.Linear(300, 256)
        self.fc2 = tnn.Linear(256, 128)
        self.fc3 = tnn.Linear(128, 64)
        self.fc4 = tnn.Linear(64, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        x = tnn.utils.rnn.pack_padded_sequence(input, lengths=length, batch_first=True)
        o, h = self.gru(x)
        x = h.transpose(0, 1)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = tnn.functional.relu(self.fc1(x))
        x = tnn.functional.relu(self.fc2(x))
        x = tnn.functional.relu(self.fc3(x))
        x = self.fc4(x)

        return x.squeeze()


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        z = []

        for i in range(len(x)):
            if x[i].find('>') != -1:
                x[i] = x[i][x[i].find('>') + 1:]

            if x[i].find('<') != -1:
                x[i] = x[i][:x[i].find('<')]

            c = x[i].find(')')
            if c != -1:
                if x[i][c - 1] in ('-', ':', ';'):
                    x[i] = 'smile'

            c = x[i].find('(')
            if c != -1:
                if x[i][c - 1] in ('-', ':', ';'):
                    x[i] = 'sad'

            c = x[i].find('d')
            if c != -1:
                if x[i][c - 1] in ('-', ':', ';'):
                    x[i] = 'laugh'

            c = x[i].find('p')
            if c != -1:
                if x[i][c - 1] in ('-', ':', ';'):
                    x[i] = 'amused'

            x[i] = ''.join(filter(lambda c: c not in "0123456789@#$^()!.?,;:\'\\/", x[i]))

            if x[i] in ('', 'is', 'a', 'his', 'movie', 'and', 'the', 'in', 'are', 'have', 'it', 'has', 'that', 'i', 'you', 'to', 'for', 'all', 'he', 'as', 'of', 'or', 'film', 'by', 'be', 'this', 'who', 'on', 'was', 'with', 'just', 'from', 'at', 'they', 'an', 'so'):
                continue

            z.append(x[i])

        return z

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()
