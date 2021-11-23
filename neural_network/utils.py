import time
import numpy as np
import random

import matplotlib.pyplot as plt


def load_csv(path):
    '''
    Load the CSV form of MNIST data without any external library
    :param path: the path of the csv file
    :return:
        data: A list of list where each sub-list with 28x28 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    '''
    data = []
    labels = []
    with open(path, 'r') as fp:
        images = fp.readlines()
        images = [img.rstrip() for img in images]

        for img in images:
            img_as_list = img.split(',')
            y = int(img_as_list[0]) # first entry as label
            x = img_as_list[1:]
            x = [int(px) / 255 for px in x]
            data.append(x)
            labels.append(y)
    return data, labels


def load_mnist_trainval():
    """
    Load MNIST training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv('./data/mnist_train.csv')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    # split training/validation data
    train_data = None
    train_label = None
    val_data = None
    val_label = None
    #############################################################################
    #    1) Split the entire training set to training data and validation       #
    #       data. Use 80% of your data for training and 20% of your data for    #
    #       validation                                                          #
    #############################################################################
    # t = random.sample(range(0, len(data)),k=int(0.8*len(data)))
    train_data = data[0:int(len(data)*0.8)]
    val_data = data[int(len(data)*0.8):len(data)]
    train_label = label[0:int(len(label)*0.8)]
    val_label = label[int(len(label)*0.8):len(label)]

    return train_data, train_label, val_data, val_label

def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    # Load training data
    print("Loading testing data...")
    data, label = load_csv('./data/mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    return data, label


def generate_batched_data(data, label, batch_size=32, shuffle=False):
    '''
    Turn raw data into batched forms
    :param data: A list of np.ndarray containing the data where each inner list contains 28x28
                 elements corresponding to pixel values in images: array([[pix1, ..., pix768], ..., [pix1, ..., pix768]])
    :param label: A list containing the batches of labels where each batch is a np.array
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :return:
        batched_data: A list whose elements are batches of images.
        batched_label: A list whose elements are batches of labels
    '''
    batched_data = None
    batched_label = None
    random.seed(1024)
    #############################################################################
    #    1) Generate batches of images with the required batch size             #
    #    2) Shuffle data and label if shuffle=True                              #
    #    It's okay if the size of your last batch is smaller than the required  #
    #    batch size                                                             #
    #############################################################################
    X = list(zip(data, label))
    if shuffle==True:
        random.shuffle(X)  # create a array with indexes for data
    data2, label2 = zip(*X)
    data2 = np.array(data2)
    label2= np.array(label2)
    q= data2.shape[0]//batch_size
    block_end = q * batch_size
    batched_data = np.split(data2[:block_end], q)
    batched_label= np.split(label2[:block_end], q)
    if data2.shape[0]%batch_size:
        batched_data = batched_data + [data2[block_end:]]
        batched_label= batched_label+ [label2[block_end:]]

    return batched_data, batched_label

def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):
    '''
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    '''
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):

        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f} \t'
                  'Batch Loss {loss:.4f}\t'
                  'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time,
                loss=loss, accuracy=accuracy))
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc



def evaluate(batched_test_data, batched_test_label, model, debug=True):
    '''
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    '''
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):

        loss, accuracy = model.forward(input, target, mode='valid')

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                  'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(batched_test_data), accuracy=accuracy))
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history,location):
    '''
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    '''
    #############################################################################
    #    1) Plot learning curves of training and validation loss                #
    #    2) Plot learning curves of training and validation accuracy            #
    #############################################################################
    plt.figure(1)
    plt.title("Loss Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss_history, label='train')
    plt.plot(valid_loss_history, label='valid')
    plt.legend(loc='upper right')
    plt.savefig("Loss Curve_{}.png".format(location.replace("/", "_")))

    plt.figure(2)
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(train_acc_history, label='train')
    plt.plot(valid_acc_history, label='valid')
    plt.legend(loc='lower right')
    plt.savefig("Accuracy_{}.png".format(location.replace("/", "_")))
