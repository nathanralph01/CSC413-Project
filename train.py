import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import BidirectionalRNNGenerator
from wrap import WrapDataset
from utils import *

output_size = glove.vectors.shape[0]
seq_length = 5


def acc(model, dataset, batch_size):
    r = 0
    c = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    for i, (in_str, target) in enumerate(loader):
        current_batch_size = in_str.size(0)
        in_str, target = in_str.to(device), target.to(device)
        hidden = model.init_hidden(current_batch_size).to(device)
        output, hidden = model(in_str, hidden)

        # Reshape output to match target's dimensions
        output = output.reshape(batch_size, seq_length, -1)
        output_last = output[:, -1, :]

        probabilities = torch.softmax(output_last, dim=-1).detach().cpu()
        next_word = torch.argmax(probabilities, dim=-1)

        # Calculate accuracy
        r += float(torch.sum(next_word == target.cpu()))
        c += target.numel()
    return r / c

def train(model, train_data, val_data, learning_rate=0.001, batch_size=100, num_epochs=10):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    seq_length = 5
    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, tloss, tacc, vacc = [], [], [], []
    count = 0 # count the number of iterations that has passed
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

            loss = criterion(output_flat, targets_flat)

            scaler.scale(loss).backward() # propagate the gradients
            scaler.step(optimizer) # update the parameters
            scaler.update()
            optimizer.zero_grad()
            count += 1

            if count % 100 == 0:
                    iters.append(count)
                    t = acc(model, train_data, batch_size)
                    v = acc(model, val_data, batch_size)
                    tloss.append(float(loss))
                    tacc.append(t)
                    vacc.append(v)
                    print(count, "Loss:", float(loss), "Training Accuracy:", t, "Validation Accuracy:", v)

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



# Run to train the model. The state dict will be saves as model_state.pth
if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    data = load_data('data/stories2.txt') # using samller txt file for testing purposes

    # convert data to sequences in order to predict next character
    seq_data = create_training_sequences(data, 5)
    # embed data
    seq_data_embed = embed_data_stories(seq_data)
    # split to train, val, test sets ->> MAY NEED TO CHANGE VAL AND TEST SETS
    train_data, val_data, test_data = split_data(seq_data_embed, 0.7, 0.15)
    # Wrap the data so that it is compatible with DataLoader
    wrapped_data = WrapDataset(train_data)
    wrapped_data_val = WrapDataset(val_data)

    model = BidirectionalRNNGenerator().to(device)

    print("Beginning training...")
    train(model, wrapped_data, wrapped_data_val, batch_size=2, num_epochs=2, learning_rate=0.0005)
    print("Training complete")
    torch.save(model.state_dict(), "model_state.pth")


