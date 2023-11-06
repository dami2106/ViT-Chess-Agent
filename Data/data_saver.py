# %%
import chess.pgn
import chess
from tqdm import tqdm
import numpy as np
import sys
import os
import gzip

# %%
ELO_THRESH = 2000
# CHUNK_SIZE = 10000000
MAX_GAMES = 400000
BLACK_WIN = MAX_GAMES//2
WHITE_WIN = MAX_GAMES//2

square_index = {
    'a' : 0,
    'b' : 1,
    'c' : 2,
    'd' : 3,
    'e' : 4,
    'f' : 5,
    'g' : 6,
    'h' : 7,
}

# %%
def square_to_index(square):
  box = chess.square_name(square)
  return (8 - int(box[1])) ,  square_index[box[0]]

def board_to_tensor(board):
  tensor_board = np.zeros((14, 8, 8), dtype = np.byte)

  for piece in chess.PIECE_TYPES:
    for square in board.pieces(piece, chess.WHITE):
      idx = np.unravel_index(square, (8, 8))
      tensor_board[piece - 1][7 - idx[0]][idx[1]] = 1
    for square in board.pieces(piece, chess.BLACK):
      idx = np.unravel_index(square, (8, 8))
      tensor_board[piece + 5][7 - idx[0]][idx[1]] = 1

  turn = board.turn
  board.turn = chess.WHITE
  for move in board.legal_moves:
    i, j = square_to_index(move.to_square)
    tensor_board[12][i][j] = 1

  board.turn = chess.BLACK
  for move in board.legal_moves:
    i, j = square_to_index(move.to_square)
    tensor_board[13][i][j] = 1

  board.turn = turn

  return tensor_board

def move_to_matrix(move):
    to = np.zeros((8,8), dtype=np.byte)
    from_ = np.zeros((8,8), dtype=np.byte)
    to[7 - move.to_square // 8, move.to_square % 8] = 1
    from_[7 - move.from_square // 8, move.from_square % 8] = 1
    return from_.flatten(), to.flatten()


# %%
# file_name = sys.argv[1]
file_name = "March_2023_2500_Elo.pgn"
file_name_no_ext = file_name.split(".")[0]
print("\nOpening {} for processing. Output will be saved to {}_formatted directory".format(file_name, file_name_no_ext))

# %%
game_count = 0
black_win_count = 0
white_win_count = 0

with open(file_name) as pgn:
    data = pgn.read()
    game_count = data.count("[Event")
    pgn.close()

print("PGN file has {} games in it.".format(game_count))

# %%
path = file_name_no_ext + "_formatted"
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Created folder '{path}'.")

# %%
boards = []
to_moves = []
from_moves = []

state_nums = 0
chunk_num = 0

# %%
with open(file_name) as pgn:
    progress_bar = tqdm(total=min(MAX_GAMES, game_count) , unit='games', desc='Saving')

    game_proc_count = 0

    while white_win_count < WHITE_WIN and black_win_count < BLACK_WIN:
        game = chess.pgn.read_game(pgn)

        if game is None:
            break
        
        if white_win_count == WHITE_WIN:
            progress_bar.update(1)
            continue
        elif black_win_count == BLACK_WIN:
            progress_bar.update(1)
            continue

        if game.headers["Result"] == '1-0':
           white_win_count += 1
        elif game.headers["Result"] == '0-1':
           black_win_count += 1

        if game.headers["WhiteElo"] == '' or game.headers["BlackElo"] == '':
          progress_bar.update(1)
          continue
        
        ELO_AVG = (int(game.headers["WhiteElo"]) + int(game.headers["BlackElo"])) / 2

        if ELO_AVG < ELO_THRESH :
          progress_bar.update(1)
          continue


        curr_board = game.board()
        for move in game.mainline_moves():

            from_, to = move_to_matrix(move)
            board = board_to_tensor(curr_board)

            #if black turn to play, then multiply by -1 so learn against it
            if not curr_board.turn:
                board *= -1

            from_moves.append(from_)
            to_moves.append(to)
            boards.append(board)

            state_nums += 1

            curr_board.push(move)
        progress_bar.update(1)
        game_proc_count += 1
    progress_bar.close()
    pgn.close()

f = gzip.GzipFile(path + "/boards.npy.gz", "w")
np.save(file=f, arr=np.array(boards, dtype=np.byte))
f.close()

f = gzip.GzipFile(path + "/to.npy.gz", "w")
np.save(file=f, arr=np.array(to_moves, dtype=np.byte))
f.close()

f = gzip.GzipFile(path + "/from.npy.gz", "w")
np.save(file=f, arr=np.array(from_moves, dtype=np.byte))
f.close()

print("Saved final chunk {} to disk".format(chunk_num))

# %%
#Create a textfile with the metadata of the format:
f = open(path + "/information.txt", "w")
f.write("Filename : {}\n".format(file_name))
f.write("{} Games saved.\n".format(game_proc_count))
f.write("{} Moves saved.\n".format(state_nums))
f.write("{} Games available in PGN.\n".format(game_count))
f.write("MAX_GAMES set to : {}\n".format(MAX_GAMES))
f.write("ELO_THRESH set to : {}\n".format(ELO_THRESH))
f.write("BLACK_WIN set to : {}\n".format(BLACK_WIN))
f.write("WHITE_WIN set to : {}\n".format(WHITE_WIN))
f.write("BLACK GAMES SAVED : {}\n".format(black_win_count))
f.write("WHITE GAMES SAVED : {}\n".format(white_win_count))
f.write("Saved to {} folder.".format(path))
f.close()
print("Done!")


