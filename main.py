from utils import *
from model import *

if __name__ == '__main__':
    try:
        while (True):
            story_prompt, reading_level_scale = fetch_input()
            # TODO: Pass the story prompt to model'
            # embed the input
            story_prompt_glove = embed_data(story_prompt)
            model = BidirectionalRNNGenerator()
            output = model(story_prompt_glove)
            print("HERE SHOULD BE A STORY!!")
    except KeyboardInterrupt:
        print("\nGoodBye!")
