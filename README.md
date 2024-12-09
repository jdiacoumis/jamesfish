# JamesFish Chess Engine

A chess engine built in Python, designed to interface with Lichess.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Unix/MacOS: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/engine.py`: Main engine logic
- `src/board.py`: Board representation and move generation
- `src/evaluation.py`: Position evaluation functions
- `tests/`: Unit tests

## Development

- Run tests: `pytest`
- Format code: `black src tests`
- Type checking: `mypy src`