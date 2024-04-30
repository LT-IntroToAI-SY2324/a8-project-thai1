from neural import NeuralNet
from sklearn.model_selection import train_test_split
from neural_net_UCI_data import parse_line, normalize

with open("abalone_data.txt", "r") as file:
    training_data = [parse_line(line.strip()) for line in file.readlines() if len(line) > 4]

training_data = normalize(training_data)

train_data, test_data = train_test_split(training_data)

network = NeuralNet(8, 5, 1)
network.train(train_data)

for i in network.test_with_expected(test_data):
    difference = round(abs(i[1][0] - i[2][0]), 3)
    # desired = i[1][0] * scale
    # actual = i[2][0] * scale
    print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")