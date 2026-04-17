# Stuff to initialise the project (and remove annoying errors)
import os
import dotenv

ENV_PATH = ".env"

ENV_CONTENTS = """
TF_ENABLE_ONEDNN_OPTS=0
"""

with open(ENV_PATH, "w") as f:
    f.write(ENV_CONTENTS)

dotenv.load_dotenv(ENV_PATH)
