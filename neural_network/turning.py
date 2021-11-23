import argparse
import yaml
import copy
import csv


from models import TwoLayerNet, SoftmaxRegression
from optimizer import SGD
from utils import load_mnist_trainval, load_mnist_test, generate_batched_data, train, evaluate, plot_curves

parser = argparse.ArgumentParser(description='nn')
parser.add_argument('--config', default='./config.yaml')

def main():
    batch_size_order, reg_order, learning_rate_order, hidden_size_order, train_acc_order, valid_acc_order, test_acc_order = run()
    out=[batch_size_order, reg_order, learning_rate_order, hidden_size_order, train_acc_order, valid_acc_order, test_acc_order]
    with open("out.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(out)

def run():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


    # Prepare MNIST data
    train_data, train_label, val_data, val_label = load_mnist_trainval()
    test_data, test_label = load_mnist_test()

    batch_size_order=[]
    reg_order = []
    learning_rate_order = []
    hidden_size_order = []
    train_acc_order=[]
    valid_acc_order=[]
    test_acc_order=[]

    for batch_size in args.batch_size:
        for reg in args.reg:
            for learning_rate in args.learning_rate:
                for hidden_size in args.hidden_size:
                    batch_size_order.append(batch_size)
                    reg_order.append(reg)
                    learning_rate_order.append(learning_rate)
                    hidden_size_order.append(hidden_size)
                    # Create a model
                    if args.type == 'SoftmaxRegression':
                        model = SoftmaxRegression()
                    elif args.type == 'TwoLayerNet':
                        model = TwoLayerNet(hidden_size=hidden_size)

                    # Optimizer
                    optimizer = SGD(learning_rate=learning_rate, reg=reg)

                    train_loss_history = []
                    train_acc_history = []
                    valid_loss_history = []
                    valid_acc_history = []
                    best_acc = 0.0
                    best_model = None
                    for epoch in range(args.epochs):

                        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label, batch_size=batch_size, shuffle=True)
                        epoch_loss, epoch_acc = train(epoch, batched_train_data, batched_train_label, model, optimizer, args.debug)

                        train_loss_history.append(epoch_loss)
                        train_acc_history.append(epoch_acc)
                        # evaluate on test data
                        batched_test_data, batched_test_label = generate_batched_data(val_data, val_label, batch_size=batch_size)
                        valid_loss, valid_acc = evaluate(batched_test_data, batched_test_label, model, args.debug)
                        if args.debug:
                            print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_acc))

                        valid_loss_history.append(valid_loss)
                        valid_acc_history.append(valid_acc)

                        if valid_acc > best_acc:
                            best_acc = valid_acc
                            best_model = copy.deepcopy(model)

                    batched_test_data, batched_test_label = generate_batched_data(test_data, test_label, batch_size=batch_size)
                    _, test_acc = evaluate(batched_test_data, batched_test_label, best_model) # test the best model
                    if args.debug:
                        print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_acc))
                    train_acc_order.append(train_acc_history[-1])
                    valid_acc_order.append(valid_acc_history[-1])
                    test_acc_order.append(test_acc)

    return batch_size_order,reg_order,learning_rate_order, hidden_size_order, train_acc_order, valid_acc_order, test_acc_order
if __name__ == '__main__':
    main()
