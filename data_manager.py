import csv
from PIL import Image
from rich import print
from network import Neural_Net

def create_img(pixels, filename = "image.png"):
    #creates image representation of a digit in dataset
    out = Image.new(mode="L", size=(28, 28))
    out.putdata(pixels)
    out.save(filename)
    out.close()

def csv_to_data(filename: str, amt = None):
    """
    converts csv data into list of objects
    returns - [{input: list[int], output: list[float]}, ...]
        input is one-hot encoded vector corresponding to label
        output is list of pixel values normalized from 0-255 to 0-1
    """
    data = []

    with open(filename, mode = 'r') as mnist:
        csvFile = csv.reader(mnist)
        for i, line in enumerate(csvFile):
            num = line[0]
            out = [0] * 10
            out[int(num)] = 1
            x = list(map(lambda a: int(a)/255, line[1:]))
            data.append({"input": x, "output": out})
            if not amt is None and amt-1 == i:
                break

    return data

def load_mnist_data():
    training_data = csv_to_data('MNIST_CSV/mnist_train.csv')    
    print(f"[green]Training data gathered. Length: {len(training_data)}")
    
    test_data = csv_to_data('MNIST_CSV/mnist_test.csv')
    print(f"[green]Test data gathered. Length: {len(test_data)}")
    return training_data, test_data

def train_with_data(model: Neural_Net, training, test, epochs: int):
    """
    training, test - lists of data formated as [{input: list[int], output: list[float]}, ...]
    """
    for _ in range(epochs):
            print("[red]Training Model")
            model.train(training)
            print("[green]Model Trained")
            score = model.evaluate(test)
            print(f"[blue]Score: {score * 100}%")
    
    while True:
        save = input("[yellow]Would you like to save model? (y/n): ")
        if save == "y":
            model.save()
            break
        elif save == "n":
            break


if __name__ == "__main__":
    # training_data, test_data = load_mnist_data()

    # nn = Neural_Net()
    # nn.load("nn_model.json")
    # score = nn.evaluate(test_data)
    # print(f"[blue]Initial Score: {score * 100}%")

    # train_with_data(nn, training_data, test_data, 1)
    pass


    



    