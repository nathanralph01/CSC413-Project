from utils import *
from model import BidirectionalRNNGenerator
from wrap import WrapDataset
import torch
from train import *

# Run to test the model on a particular input
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
if __name__ == '__main__':
    model = BidirectionalRNNGenerator().to(device)
    state_dict = torch.load("model_state.pth")
    model.load_state_dict(state_dict)
    # load state dict

    try:
        while (True):
            story_prompt, reading_level_scale = fetch_input()
            story_prompt_glove = glove_prompt = torch.tensor(embed_data(story_prompt), dtype=torch.long).to(device)
            output_glove = generate_story(model, glove_prompt)
            word_rep = fetch_word_representation_of_story(output_glove)
            print(word_rep)
    except KeyboardInterrupt:
        print("\nGoodBye!")

