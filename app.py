"""A simple Union.ai app for performing semantic search"""

import union
import os
from union import Resources, ImageSpec


# VLLM service (must be defined before the Streamlit App so we can reference
# the actual `App` object in the `dependencies` list).

from union.app.llm import VLLMApp

VLLM_IMAGE = "ghcr.io/unionai-oss/serving-vllm:0.1.17"

# Replace with your artifact URI or keep env override for flexibility
LLM_ARTIFACT = os.getenv(
    "QWEN_ARTIFACT_URI",
    "flyte://av0.2/chrismatteson/default/development/qwen-2_5-0_5B-Instruct@ap6cbd7l8dvgz965xfl4/n0/0/o0",
)

qwen_service = VLLMApp(
    name="qwen-service",
    container_image=VLLM_IMAGE,
    model=LLM_ARTIFACT,
    model_id="qwen2",
    port=8084,
    requests=Resources(cpu=4, mem="10Gi", gpu=1),
    limits=Resources(cpu=7, mem="24Gi", gpu=1),
    stream_model=False,
    extra_args="--dtype=half --enable-auto-tool-choice --tool-call-parser hermes",
    scaledown_after=500,
    requires_auth=False,
)

# ---------------------------------------------------------------------------
# Streamlit image & App
# ---------------------------------------------------------------------------

# The `ImageSpec` for the container that will run the `App`.
# `union-runtime` must be declared as a dependency,
# in addition to any other dependencies needed by the app code.
# Set the environment variable `REGISTRY` to be the URI for your container registry.
# If you are using `ghcr.io` as your registry, make sure the image is public.
image = union.ImageSpec(
    name="streamlit-app",
    apt_packages=["build-essential"],
    packages=[
        "union>=0.1.170",
        "union-runtime>=0.1.11",
        "streamlit==1.41.1",
        "vllm",
        "torch",
        "transformers",
        "chromadb==0.4.22",
        "sentence-transformers==2.7.0",
        "streamlit-autorefresh==1.0.1",  # provides st_autorefresh component
    ],
    registry=os.getenv("REGISTRY"),
)

# Define the Streamlit front-end and list the actual App object as a dependency.

app = union.app.App(
    name="semantic-search",
    container_image=image,
    args="streamlit run main.py --server.port 8080",
    port=8080,
    include=["main.py", "pages"],
    limits=Resources(cpu="2", mem="4Gi"),
    dependencies=[qwen_service],
) 