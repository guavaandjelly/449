# Import NumPy which is used for efficient numerical operations such as
# array creation and element-wise mathematics throughout the script
import numpy as np

# Each row of the triangular board contains an increasing number of
# bricks starting from 1 up to 8. ``bricks_per_row`` therefore holds
# ``[1, 2, 3, ..., 8]``.
bricks_per_row = np.arange(1, 9)

# ``bricks_cumsum`` stores the cumulative sum of ``bricks_per_row``. It
# effectively maps the index of the last brick in each row which allows
# quick conversion between a linear index and a two-dimensional
# coordinate (row, column).
bricks_cumsum = np.cumsum(bricks_per_row)


# ------------------------------
# Helper functions
# ------------------------------

# Translate a single linear index into a ``(row, column)`` coordinate on
# the triangular board.
def index_to_coord(ind):
    # ``np.where`` returns the indices where the condition is true.  The
    # first index in which ``ind`` is smaller than ``bricks_cumsum``
    # corresponds to the row that contains the brick.
    row = np.where(ind < bricks_cumsum)[0][0]

    # The column number is derived by compensating for the cumulative
    # count of previous rows. ``row + 1`` converts from zero-based column
    # indexing to the board's one-based scheme.
    col = row + 1 + ind - bricks_cumsum[row]
    return row, col


# Perform the opposite operation of ``index_to_coord``: convert a
# ``(row, column)`` coordinate back to the corresponding linear index in
# the board string.
def coord_to_index(coord):
    row, col = coord

    # ``bricks_cumsum[row]`` yields the index after the last brick of the
    # target row.  Subtracting ``row + 1`` (the number of bricks in all
    # preceding rows) and then adding ``col`` gives the zero-based index.
    return bricks_cumsum[row] - row - 1 + col


# Determine whether a brick located at ``ind`` may be removed according
# to the game rules.
def is_legal(board_list, ind):
    # Translate index to board coordinates for neighbor checks
    row, col = index_to_coord(ind)

    # A brick marked with ``*`` has already been removed
    if board_list[ind] == '*':
        return False

    # Bricks in the top row have no supporting bricks and can always be
    # removed
    if row == 0:
        return True

    # If the brick below-left exists and hasn't been removed, current
    # brick cannot be taken
    if col > 0 and board_list[coord_to_index((row - 1, col - 1))] != '*':
        return False

    # If the brick below-right exists and hasn't been removed, current
    # brick also cannot be taken
    if col < row and board_list[coord_to_index((row - 1, col))] != '*':
        return False

    # Otherwise the brick is free to remove
    return True

# Calculate the score for a set of removed bricks represented as a
# string of digits.
def compute_score(bricks_str):
    # Convert the brick characters to an integer NumPy array for easy
    # summation and counting
    arr = np.array(list(bricks_str), dtype='int')

    # Base points are the sum of the brick values themselves
    base_points = np.sum(arr)

    # Bonus points award powers of two based on how many identical bricks
    # of each value were collected
    bonus_points = np.sum(2 ** (np.unique(arr, return_counts=True)[1] - 1))

    return base_points + bonus_points

# Simple evaluation function for minimax: the difference between the
# player's score and the opponent's score.
def evaluate(board, my_bricks, opponent_bricks):
    return compute_score(my_bricks) - compute_score(opponent_bricks)

# Enumerate the indices of all removable bricks on the current board
def get_legal_moves(board_list):
    return [i for i in range(len(board_list)) if is_legal(board_list, i)]


# Recursive alpha-beta pruned minimax search that chooses the optimal
# brick to remove.
def minimax(board, my_bricks, opponent_bricks, depth, is_maximizing, alpha, beta):
    # Termination: either maximum depth reached or no bricks left
    if depth == 0 or '*' not in board:
        return evaluate(board, my_bricks, opponent_bricks), None

    # Prepare for exploring possible moves
    board_list = list(board)
    legal_moves = get_legal_moves(board_list)

    if is_maximizing:
        # Player tries to maximize his score difference
        max_eval = float('-inf')
        best_move = None
        for move in legal_moves:
            # Simulate removing the brick at ``move`` for the current player
            new_board = board[:move] + '*' + board[move + 1:]
            new_my_bricks = my_bricks + board[move]

            # Recurse assuming the opponent plays next
            eval, _ = minimax(new_board, new_my_bricks, opponent_bricks, depth - 1, False, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move

            # Alpha-beta pruning update
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        # Opponent tries to minimize our advantage
        min_eval = float('inf')
        best_move = None
        for move in legal_moves:
            # Simulate opponent removing a brick
            new_board = board[:move] + '*' + board[move + 1:]
            new_opponent_bricks = opponent_bricks + board[move]

            # Recurse back to maximizing player
            eval, _ = minimax(new_board, my_bricks, new_opponent_bricks, depth - 1, True, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move

            # Alpha-beta pruning update for minimizing player
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# Wrapper used by the external game engine to obtain our next move.
def generate_move(board, my_bricks, opponent_bricks):
    board_list = list(board)

    # Continue searching as long as there are removable bricks
    while '*' in board_list:
        # Start the minimax search with a fixed depth and extreme
        # alpha/beta values
        _, best_move = minimax(
            board,
            my_bricks,
            opponent_bricks,
            depth=10,
            is_maximizing=True,
            alpha=float('-inf'),
            beta=float('inf')
        )
        if best_move is not None:
            return index_to_coord(best_move)
    # Fallback in case no move could be determined
    return (0, 0)

'''
def generate_move2(board, my_bricks, opponent_bricks):
    board_list = list(board)
    available_ind = [i for i in range(36) if is_legal(board_list, i)]
    possible_scores = [compute_score(my_bricks + board[i]) for i in available_ind]
    move_ind = available_ind[np.argmax(possible_scores)]
    return index_to_coord(move_ind)


board = "442115171222365378312133114342625465"
my_bricks = " "
opponent_bricks = " "

# Use the generate_move function to determine the move
move = generate_move(board, my_bricks, opponent_bricks)
move2 = generate_move2(board, my_bricks, opponent_bricks)
print(move, move2)
'''


