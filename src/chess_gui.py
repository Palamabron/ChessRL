#!/usr/bin/env python

import os
import sys
import pygame
import numpy as np
import torch
import copy
import types
import argparse
from alpha_net import ChessNet
from chess_board import board as c_board
import encoder_decoder as ed
from MCTS_chess import UCT_search, do_decode_n_move_pieces

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 192, 77, 180)  # With transparency
MOVE_HIGHLIGHT = (106, 168, 79, 160)  # Lighter shade for possible moves
TEXT_COLOR = (10, 10, 10)

# Dimensions
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
MARGIN = 40
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH + 2 * MARGIN
WINDOW_HEIGHT = BOARD_SIZE + 2 * MARGIN

# Positions
BOARD_POS = (MARGIN, MARGIN)
SIDEBAR_POS = (BOARD_SIZE + MARGIN, MARGIN)

# Font
pygame.font.init()
FONT = pygame.font.Font(None, 32)
SMALL_FONT = pygame.font.Font(None, 24)

# Monkey patch the move_piece method to handle None values
def patched_move_piece(self, initial_position, final_position, promoted_piece="queen"):
    # Handle None promoted_piece
    if promoted_piece is None:
        promoted_piece = "queen"  # Default to queen if None
    
    promoted_piece = promoted_piece[:1].lower()
    if promoted_piece == "k":
        promoted_piece = "n"
    
    if self.player == 0:
        promoted = False
        i, j = initial_position
        piece = self.current_board[i,j]
        self.current_board[i,j] = " "
        i, j = final_position
        if piece == "R" and initial_position == (7,0):
            self.R1_move_count += 1
        if piece == "R" and initial_position == (7,7):
            self.R2_move_count += 1
        if piece == "K":
            self.K_move_count += 1
        x, y = initial_position
        if piece == "P":
            if abs(x-i) > 1:
                self.en_passant = j; self.en_passant_move = self.move_count
            if abs(y-j) == 1 and self.current_board[i,j] == " ": # En passant capture
                self.current_board[i+1,j] = " "
            if i == 0 and promoted_piece in ["r","b","n","q"]:
                self.current_board[i,j] = promoted_piece.upper()
                promoted = True
        if promoted == False:
            self.current_board[i,j] = piece
        self.player = 1
        self.move_count += 1

    elif self.player == 1:
        promoted = False
        i, j = initial_position
        piece = self.current_board[i,j]
        self.current_board[i,j] = " "
        i, j = final_position
        if piece == "r" and initial_position == (0,0):
            self.r1_move_count += 1
        if piece == "r" and initial_position == (0,7):
            self.r2_move_count += 1
        if piece == "k":
            self.k_move_count += 1
        x, y = initial_position
        if piece == "p":
            if abs(x-i) > 1:
                self.en_passant = j; self.en_passant_move = self.move_count
            if abs(y-j) == 1 and self.current_board[i,j] == " ": # En passant capture
                self.current_board[i-1,j] = " "
            if i == 7 and promoted_piece in ["r","b","n","q"]:
                self.current_board[i,j] = promoted_piece
                promoted = True
        if promoted == False:
            self.current_board[i,j] = piece
        self.player = 0
        self.move_count += 1

    else:
        print("Invalid move: ",initial_position,final_position,promoted_piece)


class ChessGUI:
    def __init__(self, model_path=None, mcts_simulations=800, user_plays_white=True):
        # Set up the window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("AlphaZero Chess")
        
        # Load chess pieces
        self.pieces_images = self.load_pieces()
        
        # Initialize game state
        self.board = c_board()
        self.board.move_piece = types.MethodType(patched_move_piece, self.board)
        self.selected_square = None
        self.valid_moves = []
        self.game_over = False
        self.message = "Your move (White)" if user_plays_white else "AI thinking..."
        self.move_history = []
        self.user_plays_white = user_plays_white
        self.mcts_simulations = mcts_simulations
        
        # Load AI model
        self.model = None
        if model_path:
            print(f"Loading model from {model_path}")
            self.model = self.load_model(model_path)
        else:
            print("No model specified, using random play")
            
        # If AI plays white, make its first move
        if not user_plays_white and self.model:
            self.ai_make_move()
    
    def load_model(self, model_path):
        """Load the neural network model."""
        model = ChessNet()
        
        # Check if CUDA is available
        cuda = torch.cuda.is_available()
        if cuda:
            model.cuda()
            print("Using CUDA for AI model")
        else:
            print("CUDA not available, using CPU")
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location="cuda" if cuda else "cpu")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()  # Set to evaluation mode
        
        return model
    
    def load_pieces(self):
        """Load chess piece images."""
        pieces = {}
        piece_files = {
            'P': 'w_pawn.png', 'N': 'w_knight.png', 'B': 'w_bishop.png', 
            'R': 'w_rook.png', 'Q': 'w_queen.png', 'K': 'w_king.png',
            'p': 'b_pawn.png', 'n': 'b_knight.png', 'b': 'b_bishop.png', 
            'r': 'b_rook.png', 'q': 'b_queen.png', 'k': 'b_king.png'
        }
        
        # Try to find pieces in ./assets or ./images
        for folder in ["./assets/pieces/", "./images/pieces/", "./pieces/", "./assets/", "./images/"]:
            if os.path.exists(folder):
                for piece, filename in piece_files.items():
                    path = os.path.join(folder, filename)
                    if os.path.exists(path):
                        pieces[piece] = pygame.transform.scale(
                            pygame.image.load(path), 
                            (SQUARE_SIZE, SQUARE_SIZE)
                        )
                
                # Check if all pieces were loaded
                if len(pieces) == 12:
                    return pieces
        
        # If pieces can't be found, create colored placeholders
        if len(pieces) < 12:
            print("Warning: Piece images not found, using placeholders.")
            for piece in 'PNBRQK':
                pieces[piece] = self.create_piece_placeholder(piece, WHITE)
            for piece in 'pnbrqk':
                pieces[piece] = self.create_piece_placeholder(piece, BLACK)
        
        return pieces
    
    def create_piece_placeholder(self, piece, color):
        """Create a placeholder for a piece."""
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*color, 200), (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//2.5)
        text = FONT.render(piece.upper(), True, (255-color[0], 255-color[1], 255-color[2]))
        text_rect = text.get_rect(center=(SQUARE_SIZE//2, SQUARE_SIZE//2))
        surface.blit(text, text_rect)
        return surface
    
    def draw_board(self):
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                x = BOARD_POS[0] + col * SQUARE_SIZE
                y = BOARD_POS[1] + row * SQUARE_SIZE
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Draw piece if present
                piece = self.board.current_board[row, col]
                if piece != " ":
                    self.screen.blit(self.pieces_images[piece], (x, y))
                
                # Draw coordinates
                if col == 0:
                    text = SMALL_FONT.render(f"{8-row}", True, TEXT_COLOR)
                    self.screen.blit(text, (BOARD_POS[0] - 20, y + SQUARE_SIZE//2 - 8))
                if row == 7:
                    text = SMALL_FONT.render(f"{chr(97+col)}", True, TEXT_COLOR)
                    self.screen.blit(text, (x + SQUARE_SIZE//2 - 6, BOARD_POS[1] + BOARD_SIZE + 10))
        
        # Draw board border
        pygame.draw.rect(self.screen, BLACK, (BOARD_POS[0]-2, BOARD_POS[1]-2, BOARD_SIZE+4, BOARD_SIZE+4), 2)
        
        # Highlight selected square
        if self.selected_square:
            row, col = self.selected_square
            x = BOARD_POS[0] + col * SQUARE_SIZE
            y = BOARD_POS[1] + row * SQUARE_SIZE
            highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight.fill(HIGHLIGHT)
            self.screen.blit(highlight, (x, y))
            
            # Highlight valid moves
            for move in self.valid_moves:
                r, c = move[1]  # final position
                x = BOARD_POS[0] + c * SQUARE_SIZE
                y = BOARD_POS[1] + r * SQUARE_SIZE
                move_highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                move_highlight.fill(MOVE_HIGHLIGHT)
                self.screen.blit(move_highlight, (x, y))
        
        # Draw last move highlight
        if self.move_history:
            last_move = self.move_history[-1]
            _, initial_pos, final_pos, _ = last_move
            
            for pos in [initial_pos, final_pos]:
                r, c = pos
                x = BOARD_POS[0] + c * SQUARE_SIZE
                y = BOARD_POS[1] + r * SQUARE_SIZE
                highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                highlight.fill((255, 255, 0, 100))  # Yellow with transparency
                self.screen.blit(highlight, (x, y))
    
    def draw_sidebar(self):
        """Draw the sidebar with game information."""
        x, y = SIDEBAR_POS
        
        # Draw sidebar background
        pygame.draw.rect(self.screen, (240, 240, 240), (x, y, SIDEBAR_WIDTH, BOARD_SIZE))
        pygame.draw.rect(self.screen, BLACK, (x, y, SIDEBAR_WIDTH, BOARD_SIZE), 2)
        
        # Draw game status
        title = FONT.render("AlphaZero Chess", True, BLACK)
        self.screen.blit(title, (x + 10, y + 10))
        
        # Current player
        player = "White" if self.board.player == 0 else "Black"
        text = FONT.render(f"Current player: {player}", True, BLACK)
        self.screen.blit(text, (x + 10, y + 50))
        
        # Move count
        text = FONT.render(f"Move: {self.board.move_count}", True, BLACK)
        self.screen.blit(text, (x + 10, y + 80))
        
        # Game status message
        lines = [self.message[i:i+30] for i in range(0, len(self.message), 30)]
        for i, line in enumerate(lines):
            text = SMALL_FONT.render(line, True, BLACK)
            self.screen.blit(text, (x + 10, y + 110 + i * 20))
        
        # Move history
        history_title = FONT.render("Move History:", True, BLACK)
        self.screen.blit(history_title, (x + 10, y + 180))
        
        history_start = max(0, len(self.move_history) - 10)  # Show last 10 moves
        for i, move in enumerate(self.move_history[history_start:]):
            player, initial_pos, final_pos, promotion = move
            player_str = "White" if player == 0 else "Black"
            move_text = f"{history_start+i+1}. {player_str}: {self.format_move(initial_pos, final_pos, promotion)}"
            text = SMALL_FONT.render(move_text, True, BLACK)
            self.screen.blit(text, (x + 10, y + 210 + i * 20))
        
        # Buttons
        pygame.draw.rect(self.screen, (200, 200, 200), (x + 10, y + BOARD_SIZE - 100, SIDEBAR_WIDTH - 20, 40))
        text = FONT.render("New Game", True, BLACK)
        self.screen.blit(text, (x + 20, y + BOARD_SIZE - 90))
        
        pygame.draw.rect(self.screen, (200, 200, 200), (x + 10, y + BOARD_SIZE - 50, SIDEBAR_WIDTH - 20, 40))
        text = FONT.render("Quit", True, BLACK)
        self.screen.blit(text, (x + 20, y + BOARD_SIZE - 40))
    
    def format_move(self, initial_pos, final_pos, promotion):
        """Format a move in chess notation."""
        i_row, i_col = initial_pos
        f_row, f_col = final_pos
        i_coord = f"{chr(97+i_col)}{8-i_row}"
        f_coord = f"{chr(97+f_col)}{8-f_row}"
        
        promotion_str = ""
        if promotion:
            promotion_str = f"={promotion.upper()}"
        
        return f"{i_coord}->{f_coord}{promotion_str}"
    
    def draw(self):
        """Draw the entire game window."""
        self.screen.fill(WHITE)
        self.draw_board()
        self.draw_sidebar()
        pygame.display.flip()
    
    def get_square_from_pos(self, pos):
        """Convert screen position to board coordinates."""
        x, y = pos
        board_x = x - BOARD_POS[0]
        board_y = y - BOARD_POS[1]
        
        if 0 <= board_x < BOARD_SIZE and 0 <= board_y < BOARD_SIZE:
            col = board_x // SQUARE_SIZE
            row = board_y // SQUARE_SIZE
            return (row, col)
        
        return None
    
    def get_valid_moves(self, square):
        """Get all valid moves for the selected square."""
        row, col = square
        piece = self.board.current_board[row, col]
        
        # Check if it's the correct player's turn
        if (self.user_plays_white and self.board.player == 0 and piece.isupper()) or \
           (not self.user_plays_white and self.board.player == 1 and piece.islower()):
            
            # Get all possible moves
            all_moves = self.board.actions()
            valid_moves = []
            
            for move in all_moves:
                initial_pos, final_pos, promotion = move
                if initial_pos == square:
                    valid_moves.append(move)
            
            return valid_moves
        
        return []
    
    def make_move(self, initial_pos, final_pos, promotion=None):
        """Make a move on the board."""
        # Create a copy of the board for safe handling
        board_copy = copy.deepcopy(self.board)
        board_copy.move_piece = types.MethodType(patched_move_piece, board_copy)
        
        try:
            # Apply the move
            board_copy.move_piece(initial_pos, final_pos, promotion)
            
            # Handle castling
            a, b = initial_pos
            c, d = final_pos
            piece = self.board.current_board[a, b]
            
            if piece in ["K", "k"] and abs(d-b) == 2:
                if a == 7 and d-b > 0:  # castle kingside for white
                    board_copy.player = 0
                    board_copy.move_piece((7, 7), (7, 5), None)
                if a == 7 and d-b < 0:  # castle queenside for white
                    board_copy.player = 0
                    board_copy.move_piece((7, 0), (7, 3), None)
                if a == 0 and d-b > 0:  # castle kingside for black
                    board_copy.player = 1
                    board_copy.move_piece((0, 7), (0, 5), None)
                if a == 0 and d-b < 0:  # castle queenside for black
                    board_copy.player = 1
                    board_copy.move_piece((0, 0), (0, 3), None)
            
            # Update the actual board
            self.board = board_copy
            
            # Record move in history
            self.move_history.append((self.board.player ^ 1, initial_pos, final_pos, promotion))
            
            # Check game state
            self.check_game_state()
            
            return True
        except Exception as e:
            print(f"Error making move: {e}")
            return False
    
    def check_game_state(self):
        """Check if the game is over."""
        # Check for checkmate
        if self.board.check_status() and self.board.in_check_possible_moves() == []:
            winner = "Black" if self.board.player == 0 else "White"
            self.message = f"Checkmate! {winner} wins."
            self.game_over = True
            return True
        
        # Check for stalemate
        if not self.board.check_status() and self.board.actions() == []:
            self.message = "Stalemate! Game is a draw."
            self.game_over = True
            return True
        
        # Check for threefold repetition (simplified)
        board_states = [state for player, state, _, _ in self.move_history]
        for state in board_states:
            if board_states.count(state) >= 3:
                self.message = "Draw by threefold repetition."
                self.game_over = True
                return True
        
        # Check for 50-move rule (simplified)
        if self.board.no_progress_count >= 100:  # 50 moves = 100 half-moves
            self.message = "Draw by 50-move rule."
            self.game_over = True
            return True
        
        return False
    
    def ai_make_move(self):
        """Have the AI make a move."""
        if not self.model or self.game_over:
            return
        
        self.message = "AI thinking..."
        self.draw()
        
        try:
            # Apply monkey patch to the original UCT_search function's internal board copies
            original_deepcopy = copy.deepcopy
            
            def patched_deepcopy(obj, *args, **kwargs):
                result = original_deepcopy(obj, *args, **kwargs)
                if isinstance(result, c_board):
                    result.move_piece = types.MethodType(patched_move_piece, result)
                return result
            
            # Temporarily replace copy.deepcopy with our patched version
            copy.deepcopy = patched_deepcopy
            
            try:
                # Run MCTS search to get the best move
                best_move, _ = UCT_search(self.board, self.mcts_simulations, self.model)
                
                # Decode and apply the move
                i_pos, f_pos, prom = ed.decode_action(self.board, best_move)
                initial_pos, final_pos, promotion = i_pos[0], f_pos[0], prom[0]
                
                # Make the move
                success = self.make_move(initial_pos, final_pos, promotion)
                
                if success:
                    # Clear selection
                    self.selected_square = None
                    self.valid_moves = []
                    
                    # Update message
                    if not self.game_over:
                        self.message = "Your move"
            finally:
                # Restore the original deepcopy function
                copy.deepcopy = original_deepcopy
        
        except Exception as e:
            print(f"Error in AI move: {e}")
            import traceback
            traceback.print_exc()
            self.message = "Error in AI move. Try again."
    
    def handle_click(self, pos):
        """Handle a mouse click on the board."""
        # Check if click is on the sidebar buttons
        x, y = pos
        sidebar_x, sidebar_y = SIDEBAR_POS
        
        # New Game button
        if sidebar_x + 10 <= x <= sidebar_x + SIDEBAR_WIDTH - 10 and \
           sidebar_y + BOARD_SIZE - 100 <= y <= sidebar_y + BOARD_SIZE - 60:
            self.reset_game()
            return
        
        # Quit button
        if sidebar_x + 10 <= x <= sidebar_x + SIDEBAR_WIDTH - 10 and \
           sidebar_y + BOARD_SIZE - 50 <= y <= sidebar_y + BOARD_SIZE - 10:
            pygame.quit()
            sys.exit()
        
        # Check if game is over
        if self.game_over:
            return
        
        # Check if it's user's turn
        if (self.user_plays_white and self.board.player != 0) or \
           (not self.user_plays_white and self.board.player != 1):
            return
        
        # Get the clicked square
        square = self.get_square_from_pos(pos)
        if not square:
            return
        
        # If a square is already selected
        if self.selected_square:
            # Check if the clicked square is a valid move
            move = None
            for m in self.valid_moves:
                if m[1] == square:
                    move = m
                    break
            
            if move:
                # Make the move
                initial_pos, final_pos, promotion = move
                success = self.make_move(initial_pos, final_pos, promotion)
                
                if success:
                    # Clear selection
                    self.selected_square = None
                    self.valid_moves = []
                    
                    # If game not over, let AI make a move
                    if not self.game_over and self.model:
                        self.ai_make_move()
            else:
                # Check if clicking on a different piece of the same color
                row, col = square
                piece = self.board.current_board[row, col]
                
                if (self.user_plays_white and piece.isupper()) or \
                   (not self.user_plays_white and piece.islower()):
                    self.selected_square = square
                    self.valid_moves = self.get_valid_moves(square)
                else:
                    # Deselect
                    self.selected_square = None
                    self.valid_moves = []
        else:
            # Select the square if it has a piece of the correct color
            row, col = square
            piece = self.board.current_board[row, col]
            
            if (self.user_plays_white and piece.isupper()) or \
               (not self.user_plays_white and piece.islower()):
                self.selected_square = square
                self.valid_moves = self.get_valid_moves(square)
    
    def reset_game(self):
        """Reset the game to the initial state."""
        self.board = c_board()
        self.board.move_piece = types.MethodType(patched_move_piece, self.board)
        self.selected_square = None
        self.valid_moves = []
        self.game_over = False
        self.message = "Your move (White)" if self.user_plays_white else "AI thinking..."
        self.move_history = []
        
        # If AI plays white, make its first move
        if not self.user_plays_white and self.model:
            self.ai_make_move()
    
    def run(self):
        """Main game loop."""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    self.handle_click(pos)
            
            self.draw()
            clock.tick(60)
        
        pygame.quit()


def list_available_models():
    """List all available models in the model_data directory."""
    models = []
    model_dir = "./model_data/"
    
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith(".pth.tar")]
    
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AlphaZero Chess GUI')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--mcts_sims', type=int, default=800, help='Number of MCTS simulations')
    parser.add_argument('--color', type=str, default='white', help='Play as white or black')
    parser.add_argument('--list_models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        if models:
            print("Available models:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
            print("\nUse --model with the model name to play against it.")
        else:
            print("No models found in ./model_data/ directory.")
        sys.exit()
    
    model_path = None
    if args.model:
        # Check if full path or just name
        if os.path.exists(args.model):
            model_path = args.model
        elif os.path.exists(os.path.join("./model_data/", args.model)):
            model_path = os.path.join("./model_data/", args.model)
        elif os.path.exists(os.path.join("./model_data/", args.model + ".pth.tar")):
            model_path = os.path.join("./model_data/", args.model + ".pth.tar")
        else:
            print(f"Model {args.model} not found.")
            models = list_available_models()
            if models:
                print("Available models:")
                for model in models:
                    print(f"- {model}")
            sys.exit(1)
    
    # Default to the latest trained model if none specified
    if not model_path:
        models = list_available_models()
        if models:
            # Sort by iteration number
            models.sort(key=lambda x: int(x.split("iter")[-1].split(".")[0]) if "iter" in x else 0, reverse=True)
            model_path = os.path.join("./model_data/", models[0])
            print(f"Using latest model: {models[0]}")
        else:
            print("No models found. Running without AI.")
    
    user_plays_white = args.color.lower() != "black"
    
    game = ChessGUI(model_path, args.mcts_sims, user_plays_white)
    game.run()