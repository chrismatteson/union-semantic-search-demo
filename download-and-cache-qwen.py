from pathlib import Path
from typing import Annotated
import union
import os

from flytekit import Resources, current_context, task, workflow
from flytekit.core.artifact import Artifact, Inputs
from flytekit.types.directory import FlyteDirectory

# ---------------------------------------------------------------------------
# Artifact declaration
# ---------------------------------------------------------------------------
# Creating the Artifact here lets downstream workflows / apps reference it via
#   Qwen05B.query().get(as_type=FlyteDirectory)
# or    flyte:// …/qwen-2_5-0_5B@<version>
Qwen05B = Artifact(name="qwen-2_5-0_5B-Instruct")

image = union.ImageSpec(
    builder="union",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    apt_packages=["build-essential"],
    packages=[
        "huggingface_hub",
    ],
)

# ---------------------------------------------------------------------------
# Task: download the model checkpoint from HuggingFace once and cache it.
# ---------------------------------------------------------------------------
@task(
    cache=True,
    cache_version="0.1",
    requests=Resources(mem="10Gi"),
    container_image=image,
)
def download_qwen(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Annotated[FlyteDirectory, Qwen05B]:
    """Fetch the Qwen-2.5 0.5B weights via `huggingface_hub` and return them as
    a FlyteDirectory so Union records the result as an Artifact.
    """
    from huggingface_hub import snapshot_download

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    out_dir = working_dir / "qwen_checkpoint"

    # download if not already present in the local HF cache
    snapshot_download(repo_id=model_name, local_dir=out_dir, resume_download=True)
    return FlyteDirectory(path=str(out_dir))


# ---------------------------------------------------------------------------
# Workflow: single-step – returns the same directory as the task.
# ---------------------------------------------------------------------------
@workflow
def cache_qwen_wf(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Annotated[FlyteDirectory, Qwen05B]:
    return download_qwen(model_name=model_name) 