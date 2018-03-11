import mnist_loader
import network

def main(args):
    sizes = args.get("layers", [784, 20, 10])
    epochs = args.get("epochs", 5)
    eta = args.get("eta", 3.0)
    mini_batch_size = args.get("mini_batch_size", 10)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(sizes)
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=None)
    n_test = len(test_data)
    return {"Results": "Epoch Last: {0} / {1}".format(net.evaluate(test_data), n_test)}
