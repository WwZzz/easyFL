import os, json
import gzip
import numpy as np

NAME=[]
def load_mnist(path, kind='train'):


    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def generate_dataset():

  X_train, y_train = load_mnist('raw_data/fashion', kind='train')
  X_test, y_test = load_mnist('raw_data/fashion', kind='t10k')


  # some simple normalization
  mu = np.mean(X_train.astype(np.float32), 0)
  sigma = np.std(X_train.astype(np.float32), 0)

  X_train = (X_train.astype(np.float32) - mu)/(sigma+0.001)
  X_test = (X_test.astype(np.float32) - mu)/(sigma+0.001)

  return X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist()


def main():
    train_output = "./train/mytrain.json"
    test_output = "./test/mytest.json"


    X_train, y_train, X_test, y_test = generate_dataset()


    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}


    # label 0: T-shirt(top); 2: pullover; 6: Shirt
    X_trains=[[] for i in range(10)]
    y_trains = [[] for i in range(10)]
    for idx, item in enumerate(X_train):
        i=y_train[idx]
        X_trains[i].append(X_train[idx])
        y_trains[i].append(y_train[idx])

    X_tests = [[] for i in range(10)]
    y_tests = [[] for i in range(10)]
    for idx, item in enumerate(X_test):
        i=y_test[idx]
        X_tests[i].append(X_test[idx])
        y_tests[i].append(y_test[idx])
    label_dict={0:'T-shirt', 2:'pullover', 6:'shirt'}
    selected=[0,2,6]
    cvt_labels= {}
    for i in range(len(selected)):
        cvt_labels[selected[i]]=i
    for i in selected:
        train_len=len(X_trains[i])
        print("training set for {}: {}".format(i,train_len))
        test_len = len(X_tests[i])
        uname=label_dict[i]
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_trains[i], 'y': [cvt_labels[lb] for lb in y_trains[i]]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_tests[i], 'y': [cvt_labels[lb] for lb in y_tests[i]]}
        test_data['num_samples'].append(test_len)

    with open(train_output,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()

