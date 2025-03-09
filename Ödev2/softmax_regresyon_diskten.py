import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# **Veri Setini Otomatik İndirme ve Yükleme**
transform = transforms.Compose([
    transforms.ToTensor(),  # Görüntüyü tensöre çevir
    transforms.Normalize((0.5,), (0.5,))  # Normalize et (tek kanal için)
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

batch_size = 128
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# **Model Parametreleri**
num_inputs = 784  # FashionMNIST her görüntü 28x28 boyutunda
num_outputs = 10  # FashionMNIST 10 sınıf içeriyor
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)  # Sayısal kararlılık için
    return X_exp / X_exp.sum(dim=1, keepdim=True)

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        updater()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

lr = 0.1

def updater():
    global W, b  # Global değişkenleri kullan
    with torch.no_grad():
        if W.grad is not None:
            W -= lr * W.grad
            W.grad.zero_()
        if b.grad is not None:
            b -= lr * b.grad
            b.grad.zero_()

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"Epoch {epoch + 1}: Loss={train_metrics[0]:.4f}, Train Acc={train_metrics[1]:.4f}, Test Acc={test_acc:.4f}")

num_epochs = 10  # 10 epoch ile eğitim
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=10):
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    for X, y in test_iter:
        break
    y_hat = net(X.reshape((-1, 784))).argmax(axis=1)

    fig, axes = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        img = X[i].reshape(28, 28).numpy()
        true_label = class_names[y[i].item()]
        pred_label = class_names[y_hat[i].item()]

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{true_label}\n({pred_label})", fontsize=10,
                          color=("green" if true_label == pred_label else "red"))
        axes[i].axis('off')
    plt.show()

predict_ch3(net, test_iter, n=10)  # 10 görseli göstermek için
