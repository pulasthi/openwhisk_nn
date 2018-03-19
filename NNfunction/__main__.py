import mnist_loader
import network
import time

def main(args):

    stime = time.time()*1000.0
    sizes = args.get("layers", [784, 15, 10])
    epochs = args.get("epochs", 5)
    eta = args.get("eta", 3.0)
    mini_batch_size = args.get("mini_batch_size", 10)
    rank = args.get("rank", 0)
    para = args.get("para", 1)
    dbname = args.get('dbname', 'digitnndb')
    etime = time.time()*1000.0
    print("{} - {}".format("time1", etime-stime))
    stime = time.time()*1000.0
    training_data, test_data = mnist_loader.load_data_wrapper(rank, para)
    etime = time.time()*1000.0
    print("{} - {}".format("time2", etime-stime))
    stime = time.time()*1000.0
    net = network.Network(sizes, dbname, rank, para)
    etime = time.time()*1000.0
    print("{} - {}".format("time4", etime-stime))
    stime = time.time()*1000.0
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=None)
    etime = time.time()*1000.0
    print("{} - {}".format("time5", etime-stime))
    n_test = len(test_data)
    return {"Results": "Epoch Last: {0} / {1}".format(net.evaluate(test_data), n_test)}
