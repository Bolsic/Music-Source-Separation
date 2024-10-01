from torch.utils.data import Dataset
import os
class CustomDataset(Dataset):
    def __init__(self, data_folder, gt_folder, transform=None):
        self.data_folder = data_folder
        self.gt_folder = gt_folder
        self.transform = transform

        self.data_filenames = os.listdir(data_folder)
        self.gt_filenames = os.listdir(gt_folder)
    def __len__(self):
        return len(self.data_filenames) // 2
        #return 200

    def __getitem__(self, idx):
        # Load and preprocess the input and output data
        data_filename = self.data_filenames[idx]
        gt_filename = self.gt_filenames[idx]

        data_path = os.path.join(self.data_folder, data_filename)
        gt_path = os.path.join(self.gt_folder, gt_filename)
        
        input_tensor = torch.load(data_path)
        input_tensor = torch.stack((input_tensor[0, :, :, 0], input_tensor[0, :, :, 1], input_tensor[1, :, :, 0], input_tensor[1, :, :, 1]))
        output_tensor = torch.load(gt_path)
        output_tensor = torch.stack(((output_tensor[0, :, :, 0], output_tensor[0, :, :, 1], output_tensor[1, :, :, 0], output_tensor[1, :, :, 1])))

        return input_tensor, output_tensor