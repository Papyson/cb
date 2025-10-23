
1.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv

    source .venv/bin/activate
    ```

2.  **Install the City Builder Environment:**
    *The environment needs to be installed as an editable package so the agent can import it.*
    ```bash
    pip install -e ./citybuilder_env
    ```

3.  **Install All Other Dependencies:**
    *The main `requirements.txt` file in the root directory contains all necessary packages for both the agent and the environment.*
    ```bash
    pip install -r requirements.txt
    ```

# Train for 200 episodes (default), CPU, seed 123
python -m city_builder_RL.src.main train --seed 123

# Train for 500 episodes, custom env YAML
python -m city_RL.src.main train --seed 123 --episodes 500 --cfg ./citybuilder_env/citybuilder_env/config/default.yaml

# Evaluate only (20 episodes by default)
python -m city_builder_RL.src.main eval --seed 123
