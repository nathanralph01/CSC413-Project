from utils import *
from model import *
import torch
from train import *

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
if __name__ == '__main__':
    start = time.time()
    model = BidirectionalRNNGenerator().to(device)
    train_data_embedded = embed_data_tuples(train_data[:10])
    val_data_embedded = embed_data_tuples(val_data[:10])
    print("embedded data", time.time()-start)
    train(model, train_data_embedded, val_data_embedded, batch_size=2, num_epochs=10)
    # test with there was once
    try:
        while (True):
            story_prompt, reading_level_scale = fetch_input()
            # TODO: Pass the story prompt to model'
            # embed the input
            story_prompt_glove = embed_data_alt(story_prompt)
            model = BidirectionalRNNGenerator()
            output_glove = model(story_prompt_glove)
            print(fetch_word_representation_of_story(output_glove))
            print("HERE SHOULD BE A STORY!!")
    except KeyboardInterrupt:
        print("\nGoodBye!")
