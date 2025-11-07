from pathlib import Path
import argparse
from inference_utils import (
    run_inference_original,
    run_inference_sliding_and_stitch,
    stitch_predictions_from_dir_only
)
from datasets_registry import DATASET_SETUP_FUNCS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MicroSplit LVAE inference")

    # -------------------------------------------------------------------------
    # Dataset and model setup
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PAVIA_ATN", "HAGEN", "HT_LIF24", "HT_H24"],
        help="Dataset to run inference on"
    )
    parser.add_argument("--num_channels", type=int, default=2, help="Number of channels (2, 3, or 4)")
    parser.add_argument("--exposure_duration", type=str, default="5ms", help="Exposure duration for HT_LIF24 dataset")

    # -------------------------------------------------------------------------
    # Inference behavior
    # -------------------------------------------------------------------------
    parser.add_argument("--sliding_window_flag", action="store_true",
                        help="Enable sliding-window inference")
    parser.add_argument("--stitch_only", action="store_true",
                        help="Stitch previously saved tiles only")
    parser.add_argument("--results_root", type=str, default="./Microsplit_predictions",
                        help="Where to save results")
    parser.add_argument("--raw_preds_dir", type=str, default=None,
                        help="Directory with raw .npy tiles (for stitch_only mode)")

    # -------------------------------------------------------------------------
    # Performance and configuration parameters
    # -------------------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mmse_count", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--load_pretrained_ckpt", action="store_false",
                        help="Skip pretrained model loading")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Dataset setup
    # -------------------------------------------------------------------------
    setup_fn = DATASET_SETUP_FUNCS.get(args.dataset.upper())
    if setup_fn is None:
        raise ValueError(f"❌ Unknown dataset: {args.dataset}")

    setup_kwargs = {
        "sliding_window_flag": args.sliding_window_flag
    }

    # Add dataset-specific extras
    if args.dataset.upper() == "HT_LIF24":
        setup_kwargs["num_channels"] = args.num_channels
        setup_kwargs["exposure_duration"] = args.exposure_duration

    model, experiment_config, train_dset, val_dset, test_dset = setup_fn(**setup_kwargs)


    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    if args.stitch_only:
        print("🧵 Stitching previously saved tiles only...")
        # You’ll implement this
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            results_root=args.results_root,
            batch_size=args.batch_size,
            channels=args.num_channels,
        )

    elif args.sliding_window_flag:
        print("🪟 Running sliding-window inference + stitching...")
        run_inference_sliding_and_stitch(
            model=model,
            train_dset=train_dset,
            test_dset=test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root,
            dataset=args.dataset,
        )

    else:
        print("🎯 Running standard inference...")
        run_inference_original(
            model=model,
            experiment_params=experiment_config,
            test_dset=test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mmse_count=args.mmse_count,
            grid_size=args.grid_size,
            dataset=args.dataset,
            results_root=Path(args.results_root),
        )
