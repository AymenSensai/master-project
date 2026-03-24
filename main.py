import argparse
from train import main as train_main
from evaluate import evaluate as eval_main

def main():
    parser = argparse.ArgumentParser(description="Cross-Spectral Face Recognition System")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of execution")
    
    # Train Parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    train_parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    
    # Eval Parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model on test set")
    eval_parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print(f"Starting Training Mode...")
        train_main(config_path=args.config, resume=args.resume)
    elif args.mode == "eval":
        print(f"Starting Evaluation Mode...")
        eval_main(config_path=args.config, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()
