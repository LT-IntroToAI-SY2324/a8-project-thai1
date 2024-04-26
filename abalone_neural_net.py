from neural import NeuralNet
from sklearn.model_selection import train_test_split
from neural_net_UCI_data import parse_line, normalize

with open("abalone_data.txt", "r") as file:
    training_data = [parse_line(line) for line in file.readlines() if len(line) > 4]
