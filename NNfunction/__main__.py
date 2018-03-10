import mnist_loader
import network

def main(args):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 5, 10, 3.0, test_data=None)
    n_test = len(test_data)
    return {"Results": "Epoch Last: {0} / {1}".format(net.evaluate(test_data), n_test)}
