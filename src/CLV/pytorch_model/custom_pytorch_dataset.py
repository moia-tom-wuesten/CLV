from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    """A PyTorch Dataset for handling data stored in a dictionary of features and an array of targets.

    Args:
        features_dict (dict): A dictionary of feature names as keys and feature values as values.
        targets (tensor): An array of target values.
    """

    def __init__(self, features_dict, targets):
        self.data = features_dict
        self.targets = targets

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.targets)

    def __getitem__(self, idx):
        """Returns a tuple containing the features and target for a given index.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the features and target for the given index.
            The features are stored in a dict with keys as feature names and values
            as feature values.
            Shapes  features:  (samples x sequence x feature)
                    targets:   (samples x sequence x targets)
        """
        features = {k: v[idx] for k, v in self.data.items()}
        return features, self.targets[idx]
