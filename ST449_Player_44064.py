import numpy as np

bricks_per_row = np.arange(1, 9)
bricks_cumsum = np.cumsum(bricks_per_row)


# convert index to coordinate
def index_to_coord(ind):
    row = np.where(ind < bricks_cumsum)[0][0]
    col = row + 1 + ind - bricks_cumsum[row]
    return row, col


# convert coordinate to index
def coord_to_index(coord):
    row, col = coord
    return bricks_cumsum[row] - row - 1 + col


# check if a move is legal
def is_legal(board_list, ind):
    row, col = index_to_coord(ind)
    if board_list[ind] == '*': return False
    if row == 0: return True
    if col > 0 and board_list[coord_to_index((row - 1, col - 1))] != '*': return False
    if col < row and board_list[coord_to_index((row - 1, col))] != '*': return False
    return True

def compute_score(bricks_str):
    arr = np.array(list(bricks_str), dtype='int')
    base_points = np.sum(arr)
    bonus_points = np.sum(2 ** (np.unique(arr, return_counts=True)[1] - 1))
    return base_points + bonus_points

def evaluate(board, my_bricks, opponent_bricks):
    return compute_score(my_bricks) - compute_score(opponent_bricks)

def get_legal_moves(board_list):
    return [i for i in range(len(board_list)) if is_legal(board_list, i)]

def minimax(board, my_bricks, opponent_bricks, depth, is_maximizing, alpha, beta):
    if depth == 0 or '*' not in board:
        return evaluate(board, my_bricks, opponent_bricks), None

    board_list = list(board)
    legal_moves = get_legal_moves(board_list)

    if is_maximizing:
        max_eval = float('-inf')
        best_move = None
        for move in legal_moves:
            new_board = board[:move] + '*' + board[move + 1:]
            new_my_bricks = my_bricks + board[move]
            eval, _ = minimax(new_board, new_my_bricks, opponent_bricks, depth - 1, False, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in legal_moves:
            new_board = board[:move] + '*' + board[move + 1:]
            new_opponent_bricks = opponent_bricks + board[move]
            eval, _ = minimax(new_board, my_bricks, new_opponent_bricks, depth - 1, True, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def generate_move(board, my_bricks, opponent_bricks):
    board_list = list(board)
    while '*' in board_list:
        _, best_move = minimax(board, my_bricks, opponent_bricks, depth=10, is_maximizing=True, alpha=float('-inf'), beta=float('inf'))
        if best_move is not None:
            return index_to_coord(best_move)
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


