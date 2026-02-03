import os
import sys
import subprocess
import argparse


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\nERROR in {description}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")


def main(args):
    print("\n" + "="*80)
    print("DENTAL SEGMENTATION - FULL PIPELINE")
    print("="*80)
    
    # Check if model exists
    model_exists = os.path.exists('models/best_model.pth')
    
    if not model_exists or args.retrain:
        # Step 1: Train model
        train_cmd = f"python train.py --epochs {args.epochs} --batch_size {args.batch_size}"
        run_command(train_cmd, "Model Training")
    else:
        print("\n✓ Using existing trained model")
    
    # Step 2: Generate predictions
    run_command("python predict.py", "Generating Predictions")
    
    # Step 3: Create visualizations
    run_command("python visualize.py", "Creating Visualizations")
    
    # Step 4: Calculate metrics
    run_command("python evaluate.py", "Calculating Metrics")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutputs are available in the 'outputs/' directory:")
    print("  - outputs/predictions/       : Predicted segmentation masks")
    print("  - outputs/visualizations/    : All visualizations")
    print("  - outputs/metrics/           : Evaluation metrics and reports")
    print("  - outputs/training_history.png : Training curves")
    print("\nModels saved in 'models/' directory")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full segmentation pipeline')
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain model even if one exists')
    
    args = parser.parse_args()
    main(args)