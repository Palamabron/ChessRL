# Chess Reinforcement Learning

A self-learning chess AI model built using reinforcement learning through self-play, inspired by techniques like those used in AlphaZero.

## Project Structure

```
chess_rl/
│
├── models/                  # Directory to store trained models
│
├── chess_rl/                # Package directory
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration parameters
│   ├── network.py           # Neural network architecture
│   ├── encoding.py          # Board encoding/decoding
│   ├── mcts.py              # Monte Carlo Tree Search implementation
│   ├── self_play.py         # Self-play logic
│   ├── replay_buffer.py     # Experience replay buffer
│   └── trainer.py           # Training logic
│
├── train.py                 # Main training script
├── play_gui.py              # GUI for playing against the trained model
└── README.md                # Project documentation
```

## Features

- **Reinforcement Learning**: The model learns by playing against itself.
- **Deep Residual Neural Network**: Uses a deep residual network with policy and value heads.
- **Monte Carlo Tree Search**: Explores chess positions efficiently.
- **PyGame-based GUI**: Play against the trained model with a visual interface.
- **Training Configuration**: Easily adjust training parameters.

## Requirements

- Python 3.7+
- PyTorch
- Numpy
- python-chess
- Pygame

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chess-rl.git
   cd chess-rl
   ```

2. Install the dependencies:
   ```
   pip install torch numpy python-chess pygame
   ```

## Usage

### Training the Model

To train the model from scratch:

```
python train.py
```

Training options:
- `--iterations`: Number of training iterations (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--load`: Load a saved model to continue training
- `--self-play-games`: Number of self-play games per iteration
- `--mcts-simulations`: Number of MCTS simulations per move

### Playing Against the Model

To play against a trained model using the GUI:

```
python play_gui.py --model chess_model_final.pt
```

GUI controls:
- Click to select and move pieces
- Press 'R' to reset the game
- Press 'S' to switch sides (play as black/white)

## How It Works

1. **Self-Play**: The AI generates training data by playing games against itself using MCTS.
2. **Learning**: The neural network learns from these games, improving its ability to evaluate positions and choose moves.
3. **Iteration**: As training proceeds, the AI gets stronger and generates better training data.

### Neural Network Architecture

The neural network consists of:
- Input layers that encode the chess position
- Residual blocks for deep pattern recognition
- Policy head that predicts move probabilities
- Value head that evaluates positions

### Monte Carlo Tree Search

MCTS combines:
- Exploration of new moves
- Exploitation of promising variations
- Neural network evaluations to guide the search

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation is inspired by the AlphaZero algorithm by DeepMind.
- Thanks to the python-chess library for the chess engine.
