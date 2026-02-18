from math import inf
from copy import deepcopy


def utility(board):
    for row in board:
        if all(r == row[0] != 0 for r in row):
            return row[0]

    # transpose
    cols = deepcopy(board)
    for y, row in enumerate(board):
        for x, val in enumerate(row):
            cols[x][y] = val

    for col in cols:
        if all(r == col[0] != 0 for r in col):
            return col[0]

    return 0


def terminal_test(board):
    if len(get_actions(board)) == 0:
        return True

    if utility(board) != 0:
        return True
    return False


def max_value(board):
    # print(1, board, utility(board))

    if terminal_test(board):
        return utility(board)

    max_score = inf
    for action in get_actions(board):
        score = min_value(result(board, action, 1))
        if score > max_score:
            max_score = score
    return max_score


def min_value(board):
    # print(-1, board, utility(board))

    if terminal_test(board):
        return utility(board)

    min_score = inf

    for action in get_actions(board):
        score = max_value(result(board, action, -1))
        if score < min_score:
            min_score = score
    return min_score


def get_actions(board):
    open_pos = []
    for y, rows in enumerate(board):
        for x, val in enumerate(rows):
            if val == 0:
                open_pos.append((x, y))
    return open_pos


def result(board, action, player):
    new_board = deepcopy(board)
    x, y = action
    new_board[y][x] = player
    return new_board


def minmax():
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_action = None
    best_score = -1 * inf
    for action in get_actions(board):
        score = min_value(result(board, action, 1))
        if score > best_score:
            best_action = action
            best_score = score

    print(best_score, best_action)
    return


board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
print(terminal_test(board))
board = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
print(terminal_test(board))
board = [[0, 0, -1], [0, 0, -1], [0, 0, -1]]
print(terminal_test(board))

minmax()
