import os

# transformers imports TensorFlow when it is installed, which adds about a minute to start-up
# and is never used here.
os.environ.setdefault("USE_TF", "0")
