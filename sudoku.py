"""
Sudoku app built using pygame.

Controls:
    New game: n
    Move cursor: click/arrow keys
    Guess number: [1-9]
    Add number to allowed: ^ + [1-9]
    Show allowed numbers: h
    Attempt full solve: S
    Attempt partial solve: s
    Fill square from solution: r
    Show square from solution: right click
"""

import sys
from urllib.request import urlopen

import numpy as np
import pygame


class Board(np.ndarray):
    """
    Subclass `np.ndarray` to include a bunch of useful methods.

    Information about creating the subclass is at
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(subtype, nn=3):
        obj = super(Board, subtype).__new__(
            subtype, shape=(nn ** 2, nn ** 2), dtype=int,
            buffer=None, offset=0, strides=None, order=None)
        obj.nn = nn
        obj.numbers = np.arange(1, nn ** 2 + 1, 1, dtype=int)
        obj._old = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.nn = getattr(obj, 'nn', 3)
        self.numbers = getattr(obj, 'numbers', np.arange(1, self.nn ** 2 + 1, 1))

    @property
    def allowed(self):
        if np.all(self == self._old):
            return self._old_allowed
        _allowed = np.ones((self.nn ** 2, self.nn ** 2, self.nn ** 2), dtype=bool)
        for xx in self.numbers - 1:
            for yy in self.numbers - 1:
                if self[xx, yy] > 0:
                    _allowed[xx, yy, :] = False
                else:
                    not_allowed = np.unique(np.hstack([
                        self[xx, :] - 1,
                        self[:, yy] - 1,
                        self.subsquare(xx, yy).flatten() - 1
                    ]))
                    not_allowed = not_allowed[not_allowed >= 0]
                    _allowed[xx, yy, not_allowed] = False
        self._old = self.copy()
        self._old_allowed = _allowed.copy()
        return _allowed

    def subsquare(self, xx, yy):
        return self[
               xx // self.nn * self.nn:xx // self.nn * self.nn + self.nn,
               yy // self.nn * self.nn:yy // self.nn * self.nn + self.nn
               ]

    @property
    def complete(self):
        if self.remaining == 0:
            return True
        else:
            return False

    @property
    def remaining(self):
        return np.sum(self == 0, dtype=int)

    @property
    def n_allowed_per_cell(self):
        return np.sum(self, axis=-1)

    @property
    def n_allowed_per_row(self):
        return np.sum(self.n_allowed_per_cell, axis=1)

    @property
    def n_allowed_per_column(self):
        return np.sum(self.n_allowed_per_cell, axis=0)

    def allowed_per_cell(self, xx=None, yy=None):
        if xx is not None and yy is not None:
            return self.numbers[self.allowed[xx - 1, yy - 1, :]]
        elif xx is not None:
            return [
                self.numbers[self.allowed[xx - 1, yy - 1, :]]
                for xx in self.numbers
            ]
        elif yy is not None:
            return [
                self.numbers[self.allowed[xx - 1, yy - 1, :]]
                for yy in self.numbers
            ]
        else:
            return [
                [
                    self.numbers[self.allowed[xx - 1, yy - 1, :]]
                    for xx in self.numbers
                ]
                for yy in self.numbers
            ]

    def iterate(self):
        self.current_allowed = self.allowed_per_cell()
        for xx in self.numbers - 1:
            for yy in self.numbers - 1:
                if len(self.current_allowed[xx][yy]) == 1:
                    self[yy, xx] = int(self.current_allowed[xx][yy])
        self.current_allowed = self.allowed_per_cell()
        for xx in np.arange(0, 9):
            for yy in np.arange(0, 9):
                if self[yy, xx] > 0:
                    continue
                self[yy, xx] = self.required(xx, yy)

    def solve(self, verbose=False):
        n_remaining_old = self.remaining
        while not self.complete:
            if verbose:
                print(self.remaining)
            self.iterate()
            n_remaining_new = self.remaining
            if n_remaining_new == n_remaining_old:
                print("Solver is stuck!")
                return
            else:
                n_remaining_old = n_remaining_new
        return

    def required(self, xx, yy):
        for _friends in self.friends(xx, yy):
            idxs = [
                all([
                    (kk not in self.current_allowed[ii][jj]) * 1
                    for ii, jj in _friends
                ]) for kk in self.current_allowed[xx][yy]
            ]
            if sum(idxs) == 1:
                return self.current_allowed[xx][yy][np.where(idxs)[0].squeeze()]
        return 0

    def friends(self, xx, yy):
        friends = list()
        friends.append([(xx, ii) for ii in self.numbers - 1 if not ii == yy])
        friends.append([(ii, yy) for ii in self.numbers - 1 if not ii == xx])
        _square = list()
        for ii in np.arange(xx // self.nn * self.nn, (xx // self.nn + 1) * self.nn):
            for jj in np.arange(yy // self.nn * self.nn, (yy // self.nn + 1) * self.nn):
                if ii == xx and jj == yy:
                    continue
                _square.append((ii, jj))
        friends.append(_square)
        return friends


def scrape_game():
    boards = [np.empty((9, 9), dtype=int), np.empty((9, 9), dtype=int)]
    idx = np.random.randint(0, 1000000)
    board_url = f"http://www.menneske.no/sudoku/eng/showpuzzle.html?number={idx}"
    solution_url = f"http://www.menneske.no/sudoku/eng/solution.html?number={idx}"
    for ii, url in enumerate([board_url, solution_url]):
        index = 0
        board = boards[ii]
        page = urlopen(url)
        lines = page.readlines()
        for line in lines:
            line = line.decode("utf-8")
            if "td class" in line:
                value = ''.join(filter(str.isdigit, line))
                if not value:
                    value = 0
                else:
                    value = int(value)
                board[index % 9, index // 9] = value
                index += 1
        if not index == 81:
            raise ValueError(f"Only {index} entries found")
    return boards


def draw_game(initial=False):
    screen.fill((245, 245, 220))
    if not initial:
        draw_numbers()
    for ii in range(1, 9):
        pygame.draw.line(screen, 1, [0, 90 * ii], [810, 90 * ii])
        pygame.draw.line(screen, 1, [90 * ii, 0], [90 * ii, 810])
    for ii in range(1, 3):
        pygame.draw.line(screen, 1, [0, 270 * ii], [810, 270 * ii], width=5)
        pygame.draw.line(screen, 1, [270 * ii, 0], [270 * ii, 810], width=5)
    pygame.display.update()


def draw_numbers():
    pygame.draw.rect(screen, (255, 255, 128), [x_pos * 90, y_pos * 90, 90, 90])
    for ii in range(9):
        for jj in range(9):
            if np.all(drawn_board > 0):
                if drawn_board[ii, jj] == solution[ii, jj]:
                    pygame.draw.rect(screen, (0, 128, 0), [ii * 90, jj * 90, 90, 90])
                else:
                    pygame.draw.rect(screen, (128, 0, 0), [ii * 90, jj * 90, 90, 90])
            if drawn_board[ii, jj] > 0:
                screen.blit(text_surfaces[drawn_board[ii, jj] - 1], (ii * 90 + 30, jj * 90 + 15))
            elif hint_board[ii, jj]:
                screen.blit(text_surfaces[solution[ii, jj] - 1], (ii * 90 + 30, jj * 90 + 15))
            else:
                for kk in range(9):
                    if (drawn_board.allowed[ii, jj, kk] & help_) | allowed_board[ii, jj, kk]:
                        screen.blit(small_text_surfaces[kk], (ii * 90 + kk * 10 + 2, jj * 90))


def reset():
    initial_board, solution = scrape_game()
    board = Board()
    for ii in range(9):
        for jj in range(9):
            board[ii, jj] = initial_board[ii, jj]
    allowed_board = np.zeros((9, 9, 9), dtype=bool)
    hint_board = np.zeros((9, 9), dtype=bool)
    return initial_board, solution, board, allowed_board, hint_board


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
        else:
            print(f"Unrecognized argument {sys.argv[1]}.")
        sys.exit()

    pygame.init()
    screen = pygame.display.set_mode((810, 810))
    pygame.display.set_caption("sudoku")
    draw_game(initial=True)
    run = True
    x_pos = 0
    y_pos = 0
    initial_board, solution, drawn_board, allowed_board, hint_board = reset()
    font = pygame.font.SysFont('Times New Roman', 60)
    small_font = pygame.font.SysFont('Times New Roman', 15)
    text_surfaces = [
        font.render(str(ii), True, 1) for ii in range(1, 10)
    ]
    small_text_surfaces = [
        small_font.render(str(ii), True, 1) for ii in range(1, 10)
    ]
    shift = False
    help_ = False
    draw_game()

    while run:
    # while run and np.any(drawn_board != solution):
        pygame.time.wait(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_pos = max(0, x_pos - 1)
                elif event.key == pygame.K_RIGHT:
                    x_pos = min(8, x_pos + 1)
                elif event.key == pygame.K_UP:
                    y_pos = max(0, y_pos - 1)
                elif event.key == pygame.K_DOWN:
                    y_pos = min(8, y_pos + 1)
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift = ~shift
                elif event.key == pygame.K_CAPSLOCK:
                    shift = ~shift
                elif event.key == pygame.K_r:
                    drawn_board[x_pos, y_pos] = solution[x_pos, y_pos]
                elif event.key == pygame.K_n:
                    initial_board, solution, drawn_board, allowed_board, hint_board = reset()
                elif event.key == pygame.K_x and shift:
                    pygame.quit()
                elif event.key == pygame.K_s:
                    if shift:
                        drawn_board.solve()
                    else:
                        drawn_board.iterate()
                elif event.key == pygame.K_h:
                    help_ = ~help_
                elif event.key == pygame.K_0:
                    drawn_board[x_pos, y_pos] = 0
                elif (event.key > 48) & (event.key <= 57):
                    if initial_board[x_pos, y_pos] > 0:
                        continue
                    if shift:
                        allowed_board[x_pos, y_pos, event.key - 49] = ~allowed_board[x_pos, y_pos, event.key - 49]
                    else:
                        drawn_board[x_pos, y_pos] = event.key - 48
                        allowed_board[x_pos, :, event.key - 49] = False
                        allowed_board[:, y_pos, event.key - 49] = False
                draw_game()
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift = ~shift
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.__dict__["pos"]
                x_pos = x // 90
                y_pos = y // 90
                if event.__dict__["button"] == 3:
                    hint_board[x_pos, y_pos] = True
                draw_game()
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = event.__dict__["pos"]
                x_pos = x // 90
                y_pos = y // 90
                if event.__dict__["button"] == 3:
                    hint_board[x_pos, y_pos] = False
                draw_game()

    pygame.quit()
