from torch.utils.data import Dataset
from torch import nn
import numpy as np
import pickle
import torch
import copy


def prepare_images_for_saving(images_tensor, resolution, grid_size=4, range_type="neg1pos1"):
    if range_type != "uint8":
        images_tensor = (images_tensor * 0.5 + 0.5).clamp(0, 1) * 255

    images = images_tensor[:grid_size*grid_size].permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
    grid = images.reshape(grid_size, grid_size, resolution, resolution, 3)
    grid = np.swapaxes(grid, 1, 2).reshape(grid_size*resolution, grid_size*resolution, 3)
    return grid


def cycle(dl):
    while True:
        for data in dl:
            yield data


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class EMA(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # update the parameters
        update_ema(self.ema_model.parameters(), model.parameters(), self.decay)

        # buffers are copied, not averaged
        for buffer_ema, buffer in zip(self.ema_model.buffers(), model.buffers()):
            buffer_ema.copy_(buffer)


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, shape, row_index):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    array = np.frombuffer(row_bytes, dtype=dtype)
    
    if len(shape) > 0:
        array = array.reshape(shape)
    return array


def get_array_shape_from_lmdb(lmdb_env, array_name):
    with lmdb_env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))

    return image_shape


class SDTextDataset(Dataset):
    def __init__(self, anno_path, tokenizer_one, is_sdxl=False, tokenizer_two=None):
        if anno_path.endswith(".txt"):
            self.all_prompts = []
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue 
                    else:
                        self.all_prompts.append(line)
        else:
            self.all_prompts = pickle.load(open(anno_path, "rb"))
    
        self.all_indices = list(range(len(self.all_prompts)))

        self.is_sdxl = is_sdxl # sdxl uses two tokenizers
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        print(f"Loaded {len(self.all_prompts)} prompts")

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        if prompt == None:
            prompt = ""


        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        output_dict = {
            'index': self.all_indices[idx],
            'key': prompt,
            'text_input_ids_one': text_input_ids_one,
        }

        if self.is_sdxl:
            text_input_ids_two = self.tokenizer_two(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output_dict['text_input_ids_two'] = text_input_ids_two

        return output_dict


def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)
