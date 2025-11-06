"""
Quick diagnosis script for NGSIM dataset models
"""
import torch
import numpy as np
import pandas as pd
from models import ImprovedTrajectoryTransformer
from ngsim_dataloader import make_dataloader_ngsim as make_dataloader


def diagnose_model(model_path, csv_path, n_samples=100):
    """
    Quick diagnosis of trained model on NGSIM data
    """
    print(" MODEL DIAGNOSIS")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load model
    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8
    ).to(device)
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get loader
    loader = make_dataloader(csv_path, batch_size=1, shuffle=False,
                            obs_len=10, pred_len=25, stride=10)
    
    preds, gts, obs_all = [], [], []
    print(f"\nCollecting {n_samples} predictions...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            
            target = batch['target'].to(device)
            neigh_dyn = batch['neighbors_dyn'].to(device)
            neigh_spatial = batch['neighbors_spatial'].to(device)
            lane = batch['lane'].to(device)
            gt = batch['gt'][0].cpu().numpy()
            obs = target[0].cpu().numpy()
            
            pred = model(target, neigh_dyn, neigh_spatial, lane)[0].cpu().numpy()
            
            preds.append(pred)
            gts.append(gt)
            obs_all.append(obs)
    
    preds = np.array(preds)
    gts = np.array(gts)
    obs_all = np.array(obs_all)
    
    print(f"\n Statistics on {len(preds)} samples:")
    
    # 1. Prediction variance
    pred_var = preds.var(axis=0).mean()
    gt_var = gts.var(axis=0).mean()
    var_ratio = pred_var / (gt_var + 1e-8)
    
    print(f"\n1. VARIANCE CHECK:")
    print(f"   Predicted variance: {pred_var:.6f}")
    print(f"   Ground truth variance: {gt_var:.6f}")
    print(f"   Ratio: {var_ratio:.4f}")
    
    if var_ratio < 0.1:
        print(f"   CRITICAL: Model has collapsed! Predictions are nearly constant.")
        status = "COLLAPSED"
    elif var_ratio < 0.5:
        print(f"   WARNING: Low diversity in predictions")
        status = "LOW_DIVERSITY"
    else:
        print(f"  Variance looks healthy")
        status = "HEALTHY"
    
    # 2. Mean prediction magnitude
    pred_mean_disp = np.linalg.norm(preds, axis=-1).mean()
    gt_mean_disp = np.linalg.norm(gts, axis=-1).mean()
    
    print(f"\n2. DISPLACEMENT MAGNITUDE:")
    print(f"   Predicted mean displacement: {pred_mean_disp:.4f} m")
    print(f"   Ground truth mean displacement: {gt_mean_disp:.4f} m")
    print(f"   Ratio: {pred_mean_disp / (gt_mean_disp + 1e-8):.4f}")
    
    if pred_mean_disp < 1.0:
        print(f"   Predictions are very small (model predicting minimal motion)")
    
    # 3. Direction check
    pred_final = preds[:, -1, :]
    gt_final = gts[:, -1, :]
    
    pred_angles = np.arctan2(pred_final[:, 1], pred_final[:, 0])
    gt_angles = np.arctan2(gt_final[:, 1], gt_final[:, 0])
    
    angle_std_pred = np.std(pred_angles)
    angle_std_gt = np.std(gt_angles)
    
    print(f"\n3. DIRECTION DIVERSITY:")
    print(f"   Predicted angle std: {np.degrees(angle_std_pred):.2f}°")
    print(f"   Ground truth angle std: {np.degrees(angle_std_gt):.2f}°")
    
    if angle_std_pred < 0.1:
        print(f"  All predictions point in nearly same direction!")
    else:
        print(f"  Good directional diversity")
    
    # 4. Errors
    errors = np.linalg.norm(preds - gts, axis=-1)
    ade = errors.mean()
    fde = errors[:, -1].mean()
    
    print(f"\n4. PREDICTION ERRORS:")
    print(f"   ADE (Average Displacement Error): {ade:.4f} m")
    print(f"   FDE (Final Displacement Error): {fde:.4f} m")
    
    # 5. Per-timestep error progression
    per_step_error = errors.mean(axis=0)
    error_growth = per_step_error[-1] / (per_step_error[0] + 1e-8)
    print(f"\n5. ERROR PROGRESSION:")
    print(f"   First step error: {per_step_error[0]:.4f} m")
    print(f"   Last step error: {per_step_error[-1]:.4f} m")
    print(f"   Growth factor: {error_growth:.2f}x")
    
    # 6. Velocity consistency
    pred_vel = np.linalg.norm(np.diff(preds, axis=1), axis=-1).mean()
    gt_vel = np.linalg.norm(np.diff(gts, axis=1), axis=-1).mean()
    obs_vel = np.linalg.norm(np.diff(obs_all[:, :, :2], axis=1), axis=-1).mean()
    
    print(f"\n6. VELOCITY MAGNITUDE:")
    print(f"   Observed velocity: {obs_vel:.4f} m/step")
    print(f"   Predicted velocity: {pred_vel:.4f} m/step")
    print(f"   Ground truth velocity: {gt_vel:.4f} m/step")
    
    # Final diagnosis

    print(f" DIAGNOSIS: {status}")

    if status == "COLLAPSED":
        print(f"\n  MODEL HAS COLLAPSED - Urgent fixes needed:\n")
        print(f"  1.  Model is predicting nearly constant values")
        print(f"  2.  Add diversity loss to training")
        print(f"  3.  Increase learning rate slightly (try 7e-4)")
        print(f"  4.  Reduce weight_decay to 1e-5")
        print(f"  5.  Add noise augmentation to inputs")
        print(f"  6.  Check if gradients are flowing properly")
        print(f"\n   CANNOT PROCEED TO DEPLOYMENT - Retrain first!")
        
    elif status == "LOW_DIVERSITY":
        print(f"\n  LOW DIVERSITY - Improvements recommended:\n")
        print(f"  1.  Retrain with improved loss function")
        print(f"  2.  Increase model capacity (more layers/heads)")
        print(f"  3.  Add data augmentation")
        print(f"  4.  Check if training data has sufficient diversity")
        print(f"\n    Can proceed cautiously, but expect issues with complex maneuvers")
        
    else:
        print(f"\n MODEL LOOKS HEALTHY\n")
        print(f"  Recommendations:")
        print(f"  1.  Test on different road sections")
        print(f"  2. Evaluate on lane-change specific samples")
        print(f"  3. Check performance on curved vs straight trajectories")
        print(f"  4.  Validate on longer prediction horizons")
        print(f"\n  Ready to proceed to deployment!")
    

    if ade < 0.5 and var_ratio < 0.3:
        print(f"\n WARNING: Very low ADE but also low variance!")
        print(f"   This suggests the model might be predicting simple straight lines.")
        print(f"   Check the visualizations to confirm predictions aren't trivial.")
    
    return {
        'status': status,
        'variance_ratio': var_ratio,
        'ade': ade,
        'fde': fde,
        'pred_var': pred_var,
        'gt_var': gt_var,
        'pred_displacement': pred_mean_disp,
        'gt_displacement': gt_mean_disp,
        'angle_diversity': angle_std_pred,
        'error_growth': error_growth
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python diagnose_ngsim.py path/to/model.pt path/to/US101_cleaned.csv")
        sys.exit(1)
    
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    
    results = diagnose_model(model_path, csv_path, n_samples=200)
    

    print("SUMMARY")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")