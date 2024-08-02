import csv
from PIL import Image
from rich import print
from network import Neural_Net

def create_img(pixels, filename = "image.png"):
    out = Image.new(mode="L", size=(28, 28))
    out.putdata(pixels)
    out.save(filename)
    out.close()

def data_manager(filename, amt = None):
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
    training_data = data_manager('MNIST_CSV/mnist_train.csv')    
    print(f"[green]Training data gathered. Length: {len(training_data)}")
    
    test_data = data_manager('MNIST_CSV/mnist_test.csv')
    print(f"[green]Test data gathered. Length: {len(test_data)}")
    return training_data, test_data


if __name__ == "__main__":
    training_data, test_data = load_mnist_data()

    # Accuracy with (784, 10, 10) - 86.27%
    # Accuracy with (784, 64, 10) - 88.18%
    # Accuracy with (784, 64, 36, 10) - 26.91% ???

    # nn = Neural_Net(784, 10, 10)
    # score = nn.evaluate(test_data)
    # print(f"[blue]Initial Score: {round(score, 3) * 100}%")

    nn = Neural_Net()
    nn.load("nn_model.json")
    score = nn.evaluate(test_data)
    print(f"[blue]Initial Score: {score * 100}%")


    # for _ in range(3):
    #     print("[red]Training Model")
    #     nn.train_w_batches(training_data, 10)
    #     print("[green]Model Trained")
    #     score = nn.evaluate(test_data)
    #     print(f"[blue]Score: {score * 100}%")
    #     nn.save()

    for _ in range(1):
        print("[red]Training Model")
        nn.train(training_data)
        print("[green]Model Trained")
        score = nn.evaluate(test_data)
        print(f"[blue]Score: {score * 100}%")
        # nn.save()
    
    



    