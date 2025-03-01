"""
Functions for encoding chess boards and moves
"""

import chess
import numpy as np
import torch

def encode_board(board):
    """
    Encode a chess board position into a format suitable for the neural network.
    
    Args:
        board (chess.Board): The chess board to encode
        
    Returns:
        torch.FloatTensor: Encoded board as a tensor of shape (12, 8, 8)
    """
    # 12 pieces (6 white, 6 black)
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Piece placement and types
    piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            planes[piece_idx[piece.symbol()]][rank][file] = 1
    
    # Return as a PyTorch tensor
    return torch.FloatTensor(planes)

def move_to_index(move):
    """
    Convert a chess.Move to an index for the policy vector.
    
    Args:
        move (chess.Move): The move to convert
        
    Returns:
        int: The index in the policy vector
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Handle promotions
    if move.promotion:
        # Map promotions to specific indices
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }
        return 64*64 + from_square*64 + to_square + promotion_offset[move.promotion]
    
    return from_square*64 + to_square

def index_to_move(index, board):
    """
    Convert an index from the policy vector to a chess.Move.
    
    Args:
        index (int): The index in the policy vector
        board (chess.Board): The current board state
        
    Returns:
        chess.Move: The corresponding move
    """
    # If it's a promotion move
    if index >= 64*64:
        prom_index = index - 64*64
        from_square = prom_index // 64 // 4
        to_square = (prom_index // 4) % 64
        promotion_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][prom_index % 4]
        return chess.Move(from_square, to_square, promotion_piece)
    
    # Regular move
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)
