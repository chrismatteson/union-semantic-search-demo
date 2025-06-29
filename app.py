"""A simple Union.ai app for performing semantic search"""

import union
import os

# The `ImageSpec` for the container that will run the `App`.
# `union-runtime` must be declared as a dependency,
# in addition to any other dependencies needed by the app code.
# Set the environment variable `REGISTRY` to be the URI for your container registry.
# If you are using `ghcr.io` as your registry, make sure the image is public.
image = union.ImageSpec(
    name="streamlit-app",
    packages=["union-runtime>=0.1.11", "streamlit==1.41.1"],
    registry=os.getenv("REGISTRY"),
)

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# In this case we do not need to supply any app code
# as we are using the built-in Streamlit `hello` app.
app = union.app.App(
    name="semantic-search",
    container_image=image,
    args="streamlit main.py --server.port 8080",
    port=8080,
    include=["main.py"],
    limits=union.Resources(cpu="1", mem="1Gi"),
)