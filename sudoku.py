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
        if obj is None:
            return
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
        for xx in self.numbers - 1:
            for yy in self.numbers - 1:
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
                return False
            else:
                n_remaining_old = n_remaining_new
        return True

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


def draw_game(initial=False):
    screen.fill((245, 245, 220))
    if not initial:
        draw_numbers()
    for ii in range(1, BASE ** 2):
        pygame.draw.line(screen, 1, [0, SQUARESIZE * ii], [SQUARESIZE * BASE ** 2, SQUARESIZE * ii])
        pygame.draw.line(screen, 1, [SQUARESIZE * ii, 0], [SQUARESIZE * ii, SQUARESIZE * BASE ** 2])
    for ii in range(1, BASE):
        pygame.draw.line(screen, 1, [0, SQUARESIZE * BASE * ii], [SQUARESIZE * BASE ** 2, SQUARESIZE * BASE * ii],
                         width=5)
        pygame.draw.line(screen, 1, [SQUARESIZE * BASE * ii, 0], [SQUARESIZE * BASE * ii, SQUARESIZE * BASE ** 2],
                         width=5)
    pygame.display.update()


def draw_numbers():
    pygame.draw.rect(
        screen, (255, 255, 128),
        [x_pos * SQUARESIZE, y_pos * SQUARESIZE, SQUARESIZE, SQUARESIZE]
    )
    for ii in range(BASE ** 2):
        for jj in range(BASE ** 2):
            if np.all(drawn_board > 0):
                if drawn_board[ii, jj] == solution[ii, jj]:
                    pygame.draw.rect(
                        screen, (0, 128, 0),
                        [ii * SQUARESIZE, jj * SQUARESIZE, SQUARESIZE, SQUARESIZE]
                    )
                else:
                    pygame.draw.rect(
                        screen, (128, 0, 0),
                        [ii * SQUARESIZE, jj * SQUARESIZE, SQUARESIZE, SQUARESIZE]
                    )
            if initial_board[ii, jj] > 0:
                screen.blit(
                    blue_text_surfaces[drawn_board[ii, jj] - 1],
                    (ii * SQUARESIZE + SQUARESIZE // 3, jj * SQUARESIZE + SQUARESIZE // 6)
                )
            elif drawn_board[ii, jj] > 0:
                screen.blit(
                    text_surfaces[drawn_board[ii, jj] - 1],
                    (ii * SQUARESIZE + SQUARESIZE // 3, jj * SQUARESIZE + SQUARESIZE // 6)
                )
            elif hint_board[ii, jj]:
                screen.blit(
                    text_surfaces[solution[ii, jj] - 1],
                    (ii * SQUARESIZE + SQUARESIZE // 3, jj * SQUARESIZE + SQUARESIZE // 6))
            else:
                for kk in range(BASE ** 2):
                    if (drawn_board.allowed[ii, jj, kk] & help_) | allowed_board[ii, jj, kk]:
                        screen.blit(
                            small_text_surfaces[kk],
                            (ii * SQUARESIZE + kk * SQUARESIZE // (BASE ** 2 + 1) + 2, jj * SQUARESIZE)
                        )


def reset(base=3):
    initial_board, solution = new_board(base=base)
    board = Board(nn=base)
    base = base ** 2
    for ii in range(base):
        for jj in range(base):
            board[ii, jj] = initial_board[ii, jj]
    allowed_board = np.zeros((base, base, base), dtype=bool)
    hint_board = np.zeros((base, base), dtype=bool)
    return initial_board, solution, board, allowed_board, hint_board


def new_board(base=3, empty_fraction=0.75):
    """
    """
    side = base * base
    solution = False
    while not np.any(solution):
        solution = new_base()

    board = solution.copy()

    number_of_squares = side * side
    number_empty = int(number_of_squares * empty_fraction)
    for index in np.random.choice(number_of_squares, number_empty):
        board[index // side, index % side] = 0

    return board, solution


def new_base():
    """
    Start from centre square and attempt to randomly assign values outside

    TODO:
    - allow different grid sizes
    - fix upper right/lower left to not require recursion
    """

    def _allowed(ii, x_min, x_max, x_offset, y_min, y_max, y_offset):
        return np.setxor1d(numbers, np.concatenate([total[x_min:x_max, y_offset + ii % 3], total[x_offset + ii // 3, y_min:y_max], new.flatten()[:ii]]))

    def recursively_find(array, ii, func, x_min, x_max, x_offset, y_min, y_max, y_offset):
        if np.all(array > 0):
            return array
        allowed = func(ii, x_min, x_max, x_offset, y_min, y_max, y_offset)
        if len(allowed) == 0:
            return False
        np.random.shuffle(allowed)
        for num in allowed:
            array[ii // 3, ii % 3] = num
            solution = recursively_find(array, ii + 1, func, x_min, x_max, x_offset, y_min, y_max, y_offset)
            if isinstance(solution, np.ndarray):
                return solution
            elif not solution:
                continue
            else:
                if np.all(array > 0):
                    return array
        return False

    numbers = np.arange(1, 10)

    # centre
    total = np.zeros((9, 9), dtype=int)
    total[3:6, 3:6] = np.random.permutation(numbers).reshape((3, 3))
    new = np.zeros((3, 3), dtype=int)

    # centre left
    new[0] = np.random.choice(np.setxor1d(numbers, total[3, 3:6]), 3, replace=False)
    required = np.intersect1d(np.setxor1d(numbers, np.concatenate([new[0], total[4, 3:6]])), total[5, 3:6])
    new[1] = np.random.permutation(np.concatenate([required, np.random.choice(total[3, 3:6], 3 - len(required), replace=False)]))
    new[2] = np.random.permutation(np.setxor1d(numbers, new[:2].flatten()))
    total[3:6, 0:3] = new

    # upper centre
    new = np.zeros((3, 3), dtype=int)
    new[:, 0] = np.random.choice(np.setxor1d(numbers, total[3:6, 3]), 3, replace=False)
    required = np.intersect1d(np.setxor1d(numbers, np.concatenate([new[:, 0], total[3:6, 4]])), total[3:6, 5])
    new[:, 1] = np.random.permutation(np.concatenate([required, np.random.choice(total[3:6, 3], 3 - len(required), replace=False)]))
    new[:, 2] = np.random.permutation(np.setxor1d(numbers, new[:, :2].flatten()))
    total[0:3, 3:6] = new

    new = np.zeros((3, 3), dtype=int)
    new = recursively_find(array=new, ii=0, func=_allowed, x_min=3, x_max=6, x_offset=0, y_min=3, y_max=6, y_offset=0)
    if not np.all(new > 0):
        return False
    total[:3, :3] = new

    new = np.zeros((3, 3), dtype=int)
    new = recursively_find(array=new, ii=0, func=_allowed, x_min=0, x_max=6, x_offset=0, y_min=0, y_max=0, y_offset=0)
    if not np.all(new > 0):
        return False
    total[6:9, :3] = new

    new = np.zeros((3, 3), dtype=int)
    new = recursively_find(array=new, ii=0, func=_allowed, x_min=0, x_max=0, x_offset=0, y_min=0, y_max=6, y_offset=0)
    if not np.all(new > 0):
        return False
    total[:3, 6:9] = new

    new = np.zeros((3, 3), dtype=int)
    new = recursively_find(array=new, ii=0, func=_allowed, x_min=0, x_max=3, x_offset=3, y_min=0, y_max=6, y_offset=6)
    if not np.all(new > 0):
        return False
    total[3:6, 6:9] = new

    new = np.zeros((3, 3), dtype=int)
    new = recursively_find(array=new, ii=0, func=_allowed, x_min=0, x_max=6, x_offset=6, y_min=0, y_max=3, y_offset=3)
    if not np.all(new > 0):
        return False
    total[6:9, 3:6] = new

    new = np.zeros((3, 3), dtype=int)
    new = recursively_find(array=new, ii=0, func=_allowed, x_min=0, x_max=6, x_offset=6, y_min=0, y_max=6, y_offset=6)
    if not np.all(new > 0):
        return False
    total[6:9, 6:9] = new

    assert len(np.unique(total.flatten())) == 9

    return total


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
        else:
            print(f"Unrecognized argument {sys.argv[1]}.")
        sys.exit()

    pygame.init()
    BASE = 3
    SQUARESIZE = 60
    screen = pygame.display.set_mode((SQUARESIZE * BASE ** 2, SQUARESIZE * BASE ** 2))
    pygame.display.set_caption("sudoku")
    draw_game(initial=True)
    run = True
    x_pos = 0
    y_pos = 0
    initial_board, solution, drawn_board, allowed_board, hint_board = reset(base=BASE)
    font = pygame.font.SysFont('Times New Roman', SQUARESIZE * 2 // 3)
    small_font = pygame.font.SysFont('Times New Roman', SQUARESIZE // 4)
    text_surfaces = [
        font.render(f"{ii:x}", True, 1) for ii in range(1, BASE ** 2 + 1)
    ]
    blue_text_surfaces = [
        font.render(f"{ii:x}", True, (0, 0, 255)) for ii in range(1, BASE ** 2 + 1)
    ]
    small_text_surfaces = [
        small_font.render(f"{ii:x}", True, 1) for ii in range(1, BASE ** 2 + 1)
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
                    for squares in drawn_board.friends(x_pos, y_pos):
                        for square in squares:
                            allowed_board[square[0], square[1], solution[x_pos, y_pos] - 1] = False
                elif event.key == pygame.K_n:
                    initial_board, solution, drawn_board, allowed_board, hint_board = reset(base=BASE)
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
                        for squares in drawn_board.friends(x_pos, y_pos):
                            for square in squares:
                                allowed_board[square[0], square[1], event.key - 49] = False
                draw_game()
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift = ~shift
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.__dict__["pos"]
                x_pos = x // SQUARESIZE
                y_pos = y // SQUARESIZE
                if event.__dict__["button"] == 3:
                    hint_board[x_pos, y_pos] = True
                draw_game()
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = event.__dict__["pos"]
                x_pos = x // SQUARESIZE
                y_pos = y // SQUARESIZE
                if event.__dict__["button"] == 3:
                    hint_board[x_pos, y_pos] = False
                draw_game()

    pygame.quit()
