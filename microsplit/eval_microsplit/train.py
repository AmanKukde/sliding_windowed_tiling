# Training script for HAGEN, PAVIA, etc.
import platform
from pathlib import Path
import pooch
import matplotlib.pyplot as plt
from careamics.lightning import VAEModule
from pytorchlightning import Trainer
from torch.utils.data import DataLoader

from microsplit_reproducibility.configs.factory import (
    createalgorithmconfig,
    getlikelihoodconfig,
    getlossconfig,
    getmodelconfig,
    getoptimizerconfig,
    gettrainingconfig,
    getlrschedulerconfig,
)
from microsplit_reproducibility.utils.callbacks import getcallbacks
from microsplit_reproducibility.utils.io import loadcheckpointpath
from microsplit_reproducibility.datasets import createtrainvaldatasets
from microsplit_reproducibility.utils.utils import plotinputpatches

# Dataset-specific imports
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import getmicrosplitparameters
from microsplit_reproducibility.configs.data.custom_dataset_2D import getdataconfigs
from microsplit_reproducibility.datasets.custom_dataset_2D import gettrainvaldata

# Setup dataset paths
DATAPATH = Path("group/jugaman/Datasets/PAVIAATN/data")  # Change for HAGEN, etc.
NMPATH = Path("group/jugaman/Datasets/PAVIAATN/noisemodels")

NUMCHANNELS = 2
BATCHSIZE = 32
PATCHSIZE = (64, 64)
EPOCHS = 50

# Data configs
traindataconfig, valdataconfig, testdataconfig = getdataconfigs(
    imagesize=PATCHSIZE,
    numchannels=NUMCHANNELS,
)

# MicroSplit parametrization
experimentparams = getmicrosplitparameters(
    algorithm="denoisplit",
    imgsize=PATCHSIZE,
    batchsize=BATCHSIZE,
    numepochs=EPOCHS,
    multiscalecount=3,
    noisemodelpath=NMPATH,
    targetchannels=NUMCHANNELS,
)

# Create dataset
traindset, valdset, testdset, datastats = createtrainvaldatasets(
    datapath=DATAPATH,
    trainconfig=traindataconfig,
    valconfig=valdataconfig,
    testconfig=testdataconfig,
    loaddatafunc=gettrainvaldata,
)

# Configure numworkers
if platform.system() in ["Windows", "Darwin"]:
    experimentparams["numworkers"] = 0
else:
    experimentparams["numworkers"] = 3

# Create dataloaders
traindloader = DataLoader(
    traindset,
    batch_size=experimentparams["batchsize"],
    num_workers=experimentparams["numworkers"],
    shuffle=True,
)
valdloader = DataLoader(
    valdset,
    batch_size=experimentparams["batchsize"],
    num_workers=experimentparams["numworkers"],
    shuffle=False,
)

# Prepare experiment configs
experimentparams["datastats"] = datastats
lossconfig = getlossconfig(experimentparams)
modelconfig = getmodelconfig(experimentparams)
gaussianlikconfig, noisemodelconfig, nmlikconfig = getlikelihoodconfig(experimentparams)
lrschedulerconfig = getlrschedulerconfig(experimentparams)
optimizerconfig = getoptimizerconfig(experimentparams)
trainingconfig = gettrainingconfig(experimentparams)

experimentconfig = createalgorithmconfig(
    algorithm=experimentparams["algorithm"],
    lossconfig=lossconfig,
    modelconfig=modelconfig,
    gaussianlikconfig=gaussianlikconfig,
    nmconfignoisemodelconfig,
    nmlikconfignmlikconfig,
    lrschedulerconfig=lrschedulerconfig,
    optimizerconfig=optimizerconfig,
)

# Initialize model
model = VAEModule(algorithmconfig=experimentconfig)

# Load checkpoint (optional)
ckptfolder = Path("group/jugaman/Datasets/PAVIAATN/modelcheckpoints")
selectedckpt = loadcheckpointpath(str(ckptfolder), best=True)
if selectedckpt is not None:
    from microsplit_reproducibility.notebook_utils.custom_dataset_2D import loadpretrainedmodel
    loadpretrainedmodel(model, selectedckpt)

# Show some training data
plotinputpatches(dataset=traindset, numchannels=NUMCHANNELS, numsamples=3, patchsize=128)

# Train the model
trainer = Trainer(
    max_epochs=trainingconfig.numepochs,
    accelerator="gpu",
    enable_progress_bar=True,
    callbacks=getcallbacks.checkpoints,
    precision=trainingconfig.precision,
    gradient_clip_val=trainingconfig.gradient_clip_val,
    gradient_clip_algorithm=trainingconfig.gradient_clip_algorithm,
)
trainer.fit(model=model, train_dataloaders=traindloader, val_dataloaders=valdloader)

# After training, run evaluation and save results
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    getunnormalizedpredictions,
    gettarget,
    getinput,
    fullframeevaluation,
)

# Predict on validation set
stitchedpredictions = getunnormalizedpredictions(
    model,
    valdset,
    datakey=valdset.fpath.name,
    mmsecount=experimentparams["mmsecount"],
    numworkers=0,
    batchsize=8,
)

tar = gettarget(valdset)
inp = getinput(valdset).sum(-1)

# Save stitched predictions
import dill
import numpy as np
with open("results/stitched_predictions.pkl", "wb") as f:
    dill.dump(stitchedpredictions, f)
np.save("results/stitched_predictions.npy", stitchedpredictions)

# Run full frame evaluation and save metrics
metrics = fullframeevaluation(stitchedpredictions[0], tar[0], inp[0])
with open("results/metrics.txt", "w") as f:
    f.write(str(metrics))
