from pathlib import Path

import typer
import numpy as np
import tifffile
import torch
import yaml
from pytorch_lightning import Trainer
from tqdm import tqdm

from careamics.lightning import create_predict_datamodule
from careamics.lightning.lightning_module import VAEModule
from careamics.prediction_utils import convert_outputs
from careamics.config import create_hdn_configuration


def pad_image_to_minimum_size(image, min_shape=(16, 64, 64)):
    min_shape = np.array(min_shape)
    curr_shape = np.array(image.shape)
    pad_size = np.maximum(0, min_shape - curr_shape)
    
    # Calculate padding on each side
    pad_before = pad_size // 2
    pad_after = pad_size - pad_before
    
    # Pad the image
    padded_image = np.pad(
        image,
        [(p1, p2) for p1, p2 in zip(pad_before, pad_after)],
        mode='reflect'
    )
    
    return padded_image, pad_before, pad_after


def crop_to_original_shape(array, pad_before, pad_after):
    return array[pad_before[0]:array.shape[0]-pad_after[0],
                pad_before[1]:array.shape[1]-pad_after[1],
                pad_before[2]:array.shape[2]-pad_after[2]]


def run_model_prediction(model, trainer, image, mean, std):
    predict_datamodule = create_predict_datamodule(
        pred_data=image,
        data_type='array',
        axes='ZYX',
        batch_size=1,
        tile_size=(16, 64, 64),
        tile_overlap=(8, 16, 16),
        image_means=mean,
        image_stds=std,
        tta_transforms=False
    )
    
    result = trainer.predict(model, datamodule=predict_datamodule)
    
    single_preds = [p[0] for p in result]
    preds = [p[1] for p in result]
    std_preds = [p[2] for p in result]
    tile_infos = [p[3] for p in result]
    
    single_pred = convert_outputs(list(zip(single_preds, tile_infos)), tiled=True)[0].squeeze()
    pred = convert_outputs(list(zip(preds, tile_infos)), tiled=True)[0].squeeze()
    std_pred = convert_outputs(list(zip(std_preds, tile_infos)), tiled=True)[0].squeeze()
    
    return single_pred, pred, std_pred


def run_vae_inference(model, trainer, image_path, result_path, mean, std):
    image = tifffile.imread(image_path)[:, 0]
    print(f"Original image shape: {image.shape}")
    
    padded_image, pad_before, pad_after = pad_image_to_minimum_size(image)
    print(f"Padded image shape: {padded_image.shape}")
    
    single_pred, pred, std_pred = run_model_prediction(model, trainer, padded_image, mean, std)
    
    single_pred = crop_to_original_shape(single_pred, pad_before, pad_after)
    pred = crop_to_original_shape(pred, pad_before, pad_after)
    std_pred = crop_to_original_shape(std_pred, pad_before, pad_after)
    
    result = np.stack([single_pred, pred, std_pred], axis=1)
    
    tifffile.imwrite(result_path / image_path.name, result, imagej=True, metadata={'axes': 'ZCYX'})


def predict(checkpoint_path: Path, data_folder: Path, result_folder: Path, mask: str = '*.tif'):
    result_folder.mkdir(parents=True, exist_ok=True)
    inputs = sorted(data_folder.glob(mask))

    config = create_hdn_configuration(
        experiment_name="care_unet_noise_model",
        data_type="tiff",
        axes="ZYX",
        patch_size=(16, 64, 64),
        batch_size=8,
        num_epochs=5,
        encoder_conv_strides=(1, 2, 2),
        decoder_conv_strides=(1, 2, 2),
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        predict_logvar="pixelwise"
    )
    algo_config = config.algorithm_config
    algo_config.mmse_count = 15

    image_means = [2025.66723897]
    image_stds = [2135.03466266]

    lightning_module = VAEModule(algo_config)
    state = torch.load(checkpoint_path, weights_only=True)
    lightning_module.load_state_dict(state['state_dict'])

    trainer = Trainer(accelerator="gpu")

    for image_path in tqdm(inputs):
        run_vae_inference(
            model=lightning_module,
            trainer=trainer,
            image_path=image_path, 
            result_path=result_folder, 
            mean=image_means, 
            std=image_stds
        )
        


if __name__ == "__main__":
    typer.run(predict)


#     for image_path in tqdm(inputs):
#         run_vae_inference(
#             model=lightning_module,
#             trainer=trainer,
#             image_path=image_path, 
#             result_path=result_folder, 
#             mean=image_means, 
#             std=image_stds
#         )
        