#!/usr/bin/env python

import os
import argparse
import subprocess
import sys
import platform
import numpy as np
import torch

# Create necessary directories
os.makedirs("./model_data/", exist_ok=True)
os.makedirs("./datasets/iter0/", exist_ok=True)
os.makedirs("./evaluator_data/", exist_ok=True)
os.makedirs("./pieces/", exist_ok=True)

def check_cuda():
    """Check if CUDA is available and display information."""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    return cuda_available

def download_chess_pieces():
    """Download chess piece images if they don't exist."""
    if not os.path.exists("./pieces/w_pawn.png"):
        try:
            import requests
            from io import BytesIO
            from PIL import Image
            
            # Base URL for chess piece images
            base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/"
            
            # Piece filenames
            pieces = {
                'w_pawn.png': 'wP.svg', 'w_knight.png': 'wN.svg', 'w_bishop.png': 'wB.svg',
                'w_rook.png': 'wR.svg', 'w_queen.png': 'wQ.svg', 'w_king.png': 'wK.svg',
                'b_pawn.png': 'bP.svg', 'b_knight.png': 'bN.svg', 'b_bishop.png': 'bB.svg',
                'b_rook.png': 'bR.svg', 'b_queen.png': 'bQ.svg', 'b_king.png': 'bK.svg'
            }
            
            print("Downloading chess piece images...")
            for local_name, remote_name in pieces.items():
                response = requests.get(base_url + remote_name)
                if response.status_code == 200:
                    # Convert SVG to PNG using cairosvg if available
                    try:
                        import cairosvg
                        png_data = cairosvg.svg2png(bytestring=response.content, output_width=80, output_height=80)
                        with open(f"./pieces/{local_name}", 'wb') as f:
                            f.write(png_data)
                    except ImportError:
                        # Fallback to saving the SVG directly
                        with open(f"./pieces/{local_name.replace('.png', '.svg')}", 'wb') as f:
                            f.write(response.content)
                else:
                    print(f"Failed to download {remote_name}")
            
            print("Chess piece images downloaded successfully.")
        except Exception as e:
            print(f"Failed to download chess pieces: {e}")
            print("The GUI will use placeholders for pieces.")

def check_dependencies():
    """Check if all dependencies are installed."""
    required_packages = ["numpy", "torch", "pygame", "matplotlib"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing dependencies: {', '.join(missing_packages)}")
        print("Please install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def init_model():
    """Initialize a new neural network model if it doesn't exist."""
    if os.path.exists("./model_data/current_net.pth.tar"):
        print("Model already exists. Skipping initialization.")
        return True
    
    try:
        print("Initializing a new model...")
        result = subprocess.run(
            [sys.executable, "init_model.py"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error initializing model: {e}")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("init_model.py not found. Creating it...")
        create_init_model_script()
        print("Created init_model.py. Trying again...")
        return init_model()

def create_init_model_script():
    """Create the init_model.py script if it doesn't exist."""
    script = """#!/usr/bin/env python

import os
import torch
from alpha_net import ChessNet

# Initialize a new network
net = ChessNet()
print("Initialized a new network")

# Create the model_data directory if it doesn't exist
os.makedirs("./model_data/", exist_ok=True)

# Save the initialized network
torch.save(
    {'state_dict': net.state_dict()}, 
    os.path.join("./model_data/", "current_net.pth.tar")
)
print("Saved initial model to ./model_data/current_net.pth.tar")
"""
    
    with open("init_model.py", "w") as f:
        f.write(script)
    
    # Make it executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod("init_model.py", 0o755)

def run_command(command, description):
    """Run a Python script with the specified command."""
    full_command = [sys.executable] + command
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(full_command)}")
    
    try:
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Command failed with return code {return_code}")
            return False
        
        return True
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Implementation")
    
    # Main command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the AlphaZero model")
    train_parser.add_argument("--iterations", type=int, default=5, help="Number of training iterations")
    train_parser.add_argument("--games", type=int, default=10, help="Number of games per worker")
    train_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    train_parser.add_argument("--mcts_sims", type=int, default=800, help="Number of MCTS simulations per move")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per iteration")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--games", type=int, default=100, help="Total number of games to play")
    eval_parser.add_argument("--model1", type=str, help="First model to evaluate (optional)")
    eval_parser.add_argument("--model2", type=str, help="Second model to evaluate (optional)")
    eval_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    eval_parser.add_argument("--mcts_sims", type=int, default=800, help="Number of MCTS simulations per move")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play against the trained model")
    play_parser.add_argument("--model", type=str, help="Model to play against (optional)")
    play_parser.add_argument("--color", type=str, default="white", help="Play as white or black")
    play_parser.add_argument("--mcts_sims", type=int, default=800, help="Number of MCTS simulations per move")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the environment")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check CUDA
    cuda_available = check_cuda()
    if not cuda_available:
        print("Warning: CUDA is not available. Training will be slow on CPU.")
    
    # Initialize model if it doesn't exist
    if not init_model():
        print("Failed to initialize the model. Cannot continue.")
        return
    
    # Download chess pieces for GUI
    download_chess_pieces()
    
    # Handle commands
    if args.command == "train":
        train_cmd = [
            "pipeline.py",
            "--iterations", str(args.iterations),
            "--games", str(args.games),
            "--workers", str(args.workers),
            "--mcts_sims", str(args.mcts_sims),
            "--epochs", str(args.epochs)
        ]
        run_command(train_cmd, "Training AlphaZero model")
    
    elif args.command == "evaluate":
        eval_cmd = ["evaluator.py", 
                   "--games", str(args.games),
                   "--workers", str(args.workers),
                   "--mcts_sims", str(args.mcts_sims)]
        
        if args.model1:
            eval_cmd.extend(["--model1", args.model1])
        if args.model2:
            eval_cmd.extend(["--model2", args.model2])
            
        run_command(eval_cmd, "Evaluating models")
    
    elif args.command == "play":
        play_cmd = ["chess_gui.py"]
        
        if args.model:
            play_cmd.extend(["--model", args.model])
        
        play_cmd.extend(["--color", args.color, 
                         "--mcts_sims", str(args.mcts_sims)])
        
        run_command(play_cmd, "Starting chess GUI")
    
    elif args.command == "setup":
        print("Setup complete!")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()