import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import BidirectionalRNNGenerator
from utils import *
import time

def acc(model, dataset, batch_size):
    r = 0
    c = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2)
    for i, (int_str, story) in enumerate(loader):
        z = model(int_str)
        y = torch.argmax(z, axis=2)
        r += float(torch.sum((story == y).float()))
        c += story.numel()
    return r / c

def train(model, train_data, val_data, learning_rate=0.001, batch_size=100, num_epochs=10):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    #iters, train_loss, train_acc, val_acc = [], [], [], []
    tloss = []
    tacc = []
    vacc = []
    count = 0 # count the number of iterations that has passed
    iters = []
    scaler = torch.cuda.amp.GradScaler()

    for e in range(num_epochs):
        torch.cuda.empty_cache()
        for i, (in_str, story) in enumerate(train_loader):
            in_str, story = in_str.to(device), story.to(device)

            #mixed precision training
            with torch.cuda.amp.autocast():
                z = model(in_str) # TODO
                z = z.view(-1, len(glove.stoi))
                story = story.view(-1)
                loss = criterion(z, story) # TODO # (10, 300, embedding_size)

            scaler.scale(loss).backward() # propagate the gradients
            scaler.step(optimizer) # update the parameters
            scaler.update()
            optimizer.zero_grad()
            count += 1
            if count % 20 == 0:
                    iters.append(count)
                    t = acc(model, train_data, batch_size)
                    v = acc(model, val_data, batch_size)
                    tloss.append(float(loss))
                    tacc.append(t)
                    vacc.append(v)
                    print(count, "Loss:", float(loss), "Training Accuracy:", t, "Validation Accuracy:", v)

    #
    plt.figure()
    plt.plot(iters[:len(tloss)], tloss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.figure()
    plt.plot(iters[:len(tacc)], tacc)
    plt.plot(iters[:len(vacc)], vacc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])


device = torch.device("cuda" if torch.cuda.is_available else "cpu")

if __name__ == '__main__':
    start = time.time()
    model = BidirectionalRNNGenerator().to(device)
    train_data_embedded = embed_data_tuples(train_data[:10])
    val_data_embedded = embed_data_tuples(val_data[:10])
    print("embedded data", time.time() - start)
    #train(model, train_data_embedded, val_data_embedded, batch_size=2, num_epochs=1)


