def fetch_input():
    prompt = ""
    # TODO: come back to reading level after finishing the model
    reading_level = 0
    prompt_set = False
    # reading_level_set = False
    while not prompt_set:
        prompt = input("Story prompt: ")
        prompt = prompt.lower().split(sep=" ")
        prompt = [words for words in prompt if words != ""]
        if len(prompt) <= 0:
            print("You must add a prompt with at least one word. Please try again")
        else:
            prompt_set = True

    # while not reading_level_set:
    #     try:
    #         reading_level = int(input("Reading difficulty level (1-3): "))
    #         if not (1 <= reading_level <= 3):
    #             print("You must set a reading level between 1 to 3. Please try again")
    #         else:
    #             reading_level_set = True
    #     except ValueError:
    #         print("You must set a reading level between 1 to 3. Please try again")
    return prompt, reading_level

if __name__ == '__main__':
    try:
        while (True):
            story_prompt, reading_level_scale = fetch_input()
            # TODO: Pass the story prompt to model
            print("HERE SHOULD BE A STORY!!")
    except KeyboardInterrupt:
        print("\nGoodBye!")