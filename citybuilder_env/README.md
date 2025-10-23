### Setup

*First time only, run from the project root.*

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install project and dependencies
pip install -e .
```

### Run

*Run from the project root.*

```bash
# 1. Activate the virtual environment (if needed)
source .venv/bin/activate

# 2. Execute the script
python3 run.py
```

### Examples

```bash
# Run for 10 episodes with seed 42
python3 run.py --episodes 10 --seed 42

# Use the random policy
python3 run.py --policy random

# Save replay files
python3 run.py --save-replay
```