import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import BidirectionalRNNGenerator,StoryDataset
from utils import *
import time
import torch.nn.functional as F

output_size = glove.vectors.shape[0]

def acc(model, dataset, batch_size):
    r = 0
    c = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2)
    for i, (in_str, target) in enumerate(loader):
        current_batch_size = in_str.size(0)
        in_str, target = in_str.to(device), target.to(device)
        hidden = model.init_hidden(current_batch_size).to(device)
        output, hidden = model(in_str, hidden)
        
        probabilities = F.softmax(output[-1], dim=0).detach().cpu()
        top_prob, top_idx = torch.topk(probabilities, k=1)
        next_word = top_idx.numpy()[0]

        r += float(torch.sum((next_word == target).float()))
        c += target.numel()
    return r / c

def train(model, train_data, val_data, learning_rate=0.001, batch_size=100, num_epochs=10):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #print(f"Total number of batches: {len(train_loader)}", enumerate(train_loader).shape)

    seq_length = 5
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
            current_batch_size = in_str.size(0)
            in_str, story = in_str.to(device), story.to(device)
            hidden = model.init_hidden(current_batch_size).to(device)
            optimizer.zero_grad()
            output, hidden = model(in_str, hidden)

            #reshape output
            output = output.reshape(batch_size, seq_length, -1)
            output_last = output[:, -1, :]
            output_flat = output_last.reshape(-1, output_size)
            targets_flat = story.reshape(-1)

            loss = criterion(output_flat, targets_flat) # TODO # (10, 300, embedding_size)

            scaler.scale(loss).backward() # propagate the gradients
            scaler.step(optimizer) # update the parameters
            scaler.update()
            optimizer.zero_grad()
            count += 1

            if count % 5 == 0:
                    iters.append(count)
                    t = acc(model, train_data, batch_size)
                    v = acc(model, val_data, batch_size)
                    tloss.append(float(loss))
                    tacc.append(t)
                    vacc.append(v)
                    print(count, "Loss: ", float(loss))
                    print(count, "Loss:", float(loss), "Training Accuracy:", t, "Validation Accuracy:", v)

    #
    plt.figure()
    plt.plot(iters[:len(tloss)], tloss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(iters[:len(tacc)], tacc)
    plt.plot(iters[:len(vacc)], vacc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig("acc.png")


##TODO CREATE CUSTOM BATCH FUNCTION

if __name__ == '__main__':
    #start = time.time()
    model = BidirectionalRNNGenerator().to(device)
    wrapped_data = StoryDataset(train_data[:1])
    wrapped_data_val = StoryDataset(val_data[:1])
    #train_loader = torch.utils.data.DataLoader(wrapped_data, batch_size=2, shuffle=True)
    train(model, wrapped_data, wrapped_data_val, batch_size=2, num_epochs=5)
    x,t = wrapped_data[0]
    print("FROM TRAINING DATA: ")
    print(test_word_rep(x), glove.itos[t])
    x,t = x.to(device), t.to(device)
    hidden = model.init_hidden(1).to(device)
    hidden = hidden.squeeze(1)
    output = model(x, hidden)

    probabilities = F.softmax(output[-1], dim=0).detach().cpu()
    top_prob, top_idx = torch.topk(probabilities, k=1)
    next_word = top_idx.numpy()[0]
    print(test_word_rep(next_word))


