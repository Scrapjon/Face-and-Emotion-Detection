from io import StringIO
import dotenv
dotenv.load_dotenv(stream=StringIO("TF_ENABLE_ONEDNN_OPTS=0")) # bandaid solution for annoying errors, if anyone can fix this please do.

from .data_loader import *