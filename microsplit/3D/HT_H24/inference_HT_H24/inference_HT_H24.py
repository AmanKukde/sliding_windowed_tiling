import pooch
import dill
from pathlib import Path
from careamics.lvae_training.eval_utils import (
    stitch_predictions_windowed_from_dir,
    stitch_predictions_windowed
)
from usplit.core.tiff_reader import save_tiff
from microsplit_reproducibility.datasets import create_train_val_datasets


# Dataset specific imports...
from microsplit_reproducibility.configs.parameters.HT_H24 import get_microsplit_parameters
from microsplit_reproducibility.configs.data.HT_H24 import get_data_configs
from microsplit_reproducibility.datasets.HT_H24 import get_train_val_data

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import pooch
import numpy as np

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
)
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.datasets import create_train_val_datasets

from careamics.lightning import VAEModule




DATA = pooch.create(
    path="/group/jug/aman/Datasets/HT_H24/data",
    base_url="https://download.fht.org/jug/msplit/ht_h24/data",
    registry={f"ht_h24.zip": None},
)

NOISE_MODELS = pooch.create(
    path=f"/group/jug/aman/Datasets/HT_H24/noise_models/",
    base_url=f"https://download.fht.org/jug/msplit/ht_h24/noise_models/",
    registry={
        f"noise_model_Ch0.npz": None,
        f"noise_model_Ch1.npz": None,
    },
)
for fname in NOISE_MODELS.registry:
    NOISE_MODELS.fetch(fname, progressbar=True)
print('---------')
for fname in DATA.registry:
    DATA.fetch(fname, processor=pooch.Unzip(), progressbar=True)



# setting up train, validation, and test data configs
train_data_config, val_data_config, test_data_config = get_data_configs(sliding_window_flag=True)
# setting up MicroSplit parametrization
experiment_params = get_microsplit_parameters(nm_path=NOISE_MODELS.path, batch_size=32)

# start the download of required files
train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=DATA.path / f"ht_h24.zip.unzip/ht_h24",
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=val_data_config,
    load_data_func=get_train_val_data,
)

MODEL_CHECKPOINTS = pooch.create(
    path=f"/group/jug/aman/Datasets/HT_H24/pretrained_checkpoints/",
    base_url=f"https://download.fht.org/jug/msplit/ht_h24/ckpts/",
    registry={f"best.ckpt": None},
)

pretrained_model_available = False
for f in MODEL_CHECKPOINTS.registry:
    if MODEL_CHECKPOINTS.is_available(f):
        MODEL_CHECKPOINTS.fetch(f"{f}", progressbar=True)
        pretrained_model_available = True

assert pretrained_model_available, "No suitable pretrained model for your data seems to be available.\nPlease train the model using the notebook '01_train.ipynb'."

user_selected_ckpt_folder = '/group/jug/aman/Datasets/HT_H24/pretrained_checkpoints/'
ckpt_folder = user_selected_ckpt_folder
if ckpt_folder == '':
    is_ckpt_auto_selected = True
    if len(pretrained_ckpt_folders) > 0:
        ckpt_folder = pretrained_ckpt_folders[0]
    if len(ckpt_folders) > 0: # prefer to use self-trained checkpoints
        ckpt_folder = ckpt_folders[0]
else:
    is_ckpt_auto_selected = False
    
if ckpt_folder=='':
    print("🚨 CRITICAL: No model checkpoint seems to be available!")
else:
    if is_ckpt_auto_selected:
        print("⚠️ Model checkpoint to be used was automatically selected!")
    selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)
    print("✅ Selected model checkpoint:", selected_ckpt)

experiment_params["data_stats"] = data_stats

# setting up model config (using default parameters)
model_config = get_model_config(**experiment_params)

# making our data_stats known to the experiment (model) we prepare
experiment_params["data_stats"] = data_stats

# setting up model config (using default parameters)
model_config = get_model_config(**experiment_params)

# NOTE: The creation of the following configs are not strictly necessary for prediction,
#     but they ARE currently expected by the create_algorithm_config function below.
#     They act as a placeholder for now and we will work to remove them in a following release
loss_config = get_loss_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)

# finally, assemble the full set of experiment configurations...
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
)

model = VAEModule(algorithm_config=experiment_config)


# ckpt_dict = torch.load(selected_ckpt, map_location='cuda', weights_only=True)
# model.model.load_state_dict(ckpt_dict["state_dict"], strict=False)

selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)
print("✅ Selected model checkpoint:", selected_ckpt)


def get_save_dir(results_root, dataset):
    """
    Build save path as results_root/dataset/
    """
    save_dir = Path(results_root) / dataset 
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_raw_preds_dir(save_dir):
    """
    Build raw predictions folder under save_dir/raw_predictions_<exposure>_<checkpoint>
    """
    raw_dir = save_dir / f"raw_predictions_windowed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def run_inference_sliding(model, test_dset, results_root,
                        batch_size=32, num_workers=4, dataset="HT_H24"):
    """
    Run model inference on the test dataset, predict all tiles, and save them as .npy files.
    Does NOT check for existing predictions and does NOT stitch.
    
    Args:
        model: PyTorch model
        test_dset: Dataset for inference
        save_dir: Directory to save predictions (will be created if doesn't exist)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
    """
    print("Initialising")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    save_dir = get_save_dir(results_root, dataset)
    save_dir = Path(save_dir)
    # fast_delete(save_dir)  # deletes folder and all files 
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Dataloader")
    dloader = DataLoader(test_dset, pin_memory=True, num_workers=num_workers,
                         shuffle=False, batch_size=batch_size)
    print("Predicting now....")
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp = batch[0].to(device)
            rec, _ = model(inp)

            # get reconstructed img
            if model.model.predict_logvar is None:
                rec_img = rec
                logvar = torch.tensor([-1])
            else:
                rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)

            rec_np = rec_img.cpu().numpy()  # shape: (batch_size, C, H, W)
            
            for i in range(rec_np.shape[0]):
                raw_pred_path = get_raw_preds_dir(save_dir)
                pred_path =  raw_pred_path / f"pred_{global_idx:010d}.npy"
                np.save(pred_path, rec_np[i])
                global_idx += 1
                yield rec_np[i]                
    print(f"✅ Saved {global_idx} prediction tiles to {save_dir}")

def run_inference_sliding_and_stitch(
    model,
    train_dset,
    test_dset,
    batch_size=128,
    num_workers=4,
    results_root="./results_HT_H24",
    dataset="HT_H24",
):
    """
    Run sliding-window inference and stitch predictions.
    """
    prediction_generator = run_inference_sliding(
        model,
        test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        results_root=results_root,
        dataset=dataset,
    )
    save_dir = get_save_dir(results_root,dataset)

    stitched_predictions, coverage_mask = stitch_predictions_windowed(prediction_generator,test_dset,len(test_dset),inner_fraction= [1,0.5,0.5])

    mean_params, std_params = train_dset.get_mean_std()
    stitched_predictions = stitched_predictions * std_params[
        "target"
    ].squeeze().reshape(1, 1, 1, 1, -1) + mean_params["target"].squeeze().reshape(
        1, 1, 1, 1, -1
    )

    save_tiff(save_dir / "pred_test_dset_microsplit_stitched.tiff", stitched_predictions.transpose(0, 4, 1, 2, 3))
    with open(save_dir / "pred_test_dset_microsplit_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"✅ Saved stitched predictions to {save_dir}")
    return stitched_predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run sliding-window inference and stitching for HT_H24 dataset"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="./results_HT_H24",
        help="Root directory to save results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    args = parser.parse_args()


    print("🪟 Running sliding-window inference + stitching")
    run_inference_sliding_and_stitch(
        model,
        train_dset,
        test_dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        results_root=args.results_root,
        dataset="HT_H24",
    )