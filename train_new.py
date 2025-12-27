"""
Training script with BEST model checkpoint saving based on validation loss.
Works for pix2pix and CycleGAN (paired / unpaired).
"""

import os
import glob
import time
import torch
from copy import deepcopy
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


# ---------------------------------------------------
# Validation Function
# ---------------------------------------------------
# def evaluate_validation_loss(model, val_dataset):
#     """
#     Compute average validation loss.
#     For pix2pix: uses G_L1
#     For CycleGAN: uses cycle consistency loss
#     """
#     print("\n🔍 Running validation...")
#     model.eval()

#     total_loss = 0.0
#     count = 0

#     with torch.no_grad():
#         for data in val_dataset:
#             model.set_input(data)
#             model.forward()
#             losses = model.get_current_losses()

#             # Pix2Pix preferred metric
#             if "G_L1" in losses:
#                 total_loss += float(losses["G_L1"])

#             # CycleGAN preferred metric
#             elif "cycle_A" in losses and "cycle_B" in losses:
#                 total_loss += float(losses["cycle_A"] + losses["cycle_B"])

#             # Fallback: sum generator losses
#             else:
#                 total_loss += sum(float(v) for k, v in losses.items() if "G" in k)

#             count += 1

#     model.train()
#     avg_loss = total_loss / max(count, 1)
#     print(f"✅ Validation Avg Loss = {avg_loss:.4f}")
#     return avg_loss
# 

def evaluate_validation_loss(model, val_dataset):
    """
    Compute average validation loss.
    - Pix2Pix  → G_L1
    - CycleGAN → cycle_A + cycle_B
    """

    print("\n🔍 Running validation...")

    # -------------------------
    # Switch networks to eval
    # -------------------------
    if hasattr(model, "netG"):  # pix2pix
        model.netG.eval()
        model.netD.eval()
    else:  # cyclegan
        model.netG_A.eval()
        model.netG_B.eval()
        model.netD_A.eval()
        model.netD_B.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for data in val_dataset:
            model.set_input(data)
            model.forward()
            losses = model.get_current_losses()

            # ---- Pix2Pix ----
            if "G_L1" in losses:
                loss_val = losses["G_L1"]

            # ---- CycleGAN ----
            elif "cycle_A" in losses and "cycle_B" in losses:
                loss_val = losses["cycle_A"] + losses["cycle_B"]

            # ---- Fallback ----
            else:
                loss_val = sum(v for k, v in losses.items() if "G" in k)

            total_loss += float(loss_val)
            count += 1

    # -------------------------
    # Restore train mode
    # -------------------------
    if hasattr(model, "netG"):
        model.netG.train()
        model.netD.train()
    else:
        model.netG_A.train()
        model.netG_B.train()
        model.netD_A.train()
        model.netD_B.train()

    avg_loss = total_loss / max(count, 1)
    print(f"✅ Validation Avg Loss = {avg_loss:.4f}")

    return avg_loss



# ---------------------------------------------------
# Main Training Loop
# ---------------------------------------------------
if __name__ == "__main__":

    # ------------------ Options ------------------
    opt = TrainOptions().parse()
    opt.device = init_ddp()

    # ------------------ Training Dataset ------------------
    train_dataset = create_dataset(opt)
    print(f"🧩 Number of training images = {len(train_dataset)}")

    # ------------------ Validation Dataset ------------------
    val_opt = deepcopy(opt)
    val_opt.phase = "val"
    val_opt.serial_batches = True
    val_opt.isTrain = False
    val_opt.no_flip = True

    val_dataset = create_dataset(val_opt)
    print(f"🧪 Number of validation images = {len(val_dataset)}")

    # ------------------ Model ------------------
    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)

    total_iters = 0
    best_val_loss = float("inf")

    # ------------------ Epoch Loop ------------------
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        # ------------------ Training Loop ------------------
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # Display visuals
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result
                )

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            iter_data_time = time.time()

        # ------------------ Validation ------------------
        val_loss = evaluate_validation_loss(model, val_dataset)

        # ------------------ Save BEST Model ------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n🎯 New BEST model at epoch {epoch} (val loss: {val_loss:.4f})")

            # Save only from rank 0 (DDP safe)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                checkpoints_dir = os.path.join("checkpoints", opt.name)

                # Remove old best checkpoints
                for f in glob.glob(os.path.join(checkpoints_dir, "best_net_*.pth")):
                    os.remove(f)

                model.save_networks("best")
                print("💾 Best model saved\n")

        # ------------------ LR Update ------------------
        model.update_learning_rate()

        # ------------------ Regular Latest Save ------------------
        if epoch % opt.save_epoch_freq == 0:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                model.save_networks("latest")

        print(
            f"🕒 End of epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} | "
            f"Time: {time.time() - epoch_start_time:.0f}s"
        )

    cleanup_ddp()
