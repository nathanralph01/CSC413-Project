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
        story = ""
        for line in file:
            stripped = line.strip()

            if stripped == '':
                if story:
                    data_set.append(story)
                    story = ""
            else:
                story +=  " " + stripped
        data_set.append(story)
    return data_set


def split_data(data_set, train_split, val_split, test_split):
    """
    Split data into separate training, validation, and test sets.
    
    Parameters:
        data_set: Data in list format
        train_split: Percentage to split into training (0.7 for 70%)
        val_split: Percentage to split into validation
        test_split: Percentage to split into testing
    """
    total = len(data_set)
    train_num = int(total * train_split)
    val_num = train_num + int(total*val_split)

    return data_set[:train_num], data_set[train_num:val_num], data_set[val_num:]

