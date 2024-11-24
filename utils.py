import os

def load_data(path):
    """
    Load data set into a list

    Parameters:
        Path: Path of the data set
    """
    if not os.path.exists(path):
        raise Exception("File path {} does not exist".format(path))
    data_set = []
    with open(path, "r") as file:
        story = []
        for line in file:
            stripped = line.strip()

            if stripped == '':
                if story:
                    data_set.append(story)
                    story = []
            else:
                story.append(stripped)
    return data_set


def split_data(data_set, train_split, val_split, test_split):
    """
    Split data into separate training, validation, and test sets.
    
    Parameters:
        data_set: Data in list format
        train_split: Percentage to split into training
        val_split: Percentage to split into validation
        test_split: Percentage to split into testing
    """
    pass

