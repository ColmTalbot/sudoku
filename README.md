# sudoku

An interactive sudoku game built using `pygame`.

### Installation:

Currently the app itself does not install, however you will need to make sure `numpy` and `pygame` are installed.

```bash
$ pip install -r requirements.txt
```

### Running the game:

To start the game run

```bash
$ python sudoku.py
```

### Controls:

- New game: n
- Move cursor: left click/arrow keys
- Guess number: [1-9]
- Add number to allowed: ^ + [1-9]
- Show allowed numbers: h
- Attempt full solve: S
- Attempt partial solve: s
- Fill square from solution: r
- Show square from solution: right click

The solver contains limited logic and so will not always reveal the full solution.