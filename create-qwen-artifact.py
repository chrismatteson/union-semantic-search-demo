import os
import union
from flytekit.types.directory import FlyteDirectory
from flytekit import Resources
from typing_extensions import Annotated
# -------------------------------------------------------------------------------------------------
# Artifact declaration ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# Any task or workflow can later `query()` this handle to get the latest version of the model.
Qwen14BChat = union.Artifact(name="qwen-14b-chat")

# -------------------------------------------------------------------------------------------------
# Container image used for the download task ------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# Needs git-lfs (for large files) and huggingface_hub to fetch the checkpoint programmatically.
image = union.ImageSpec(
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    builder="union",
    apt_packages=["git", "git-lfs"],
    packages=[
        "huggingface_hub>=0.23",  # download HF repos programmatically
    ],
)

# -------------------------------------------------------------------------------------------------
# Task: download the model and materialise it as an artifact --------------------------------------
# -------------------------------------------------------------------------------------------------
@union.task(
    container_image=image,
    cache=True,
    cache_version="0",  # bump to force a new download / new artifact version
    requests=Resources(cpu="2", mem="8Gi"),
)
def fetch_model() -> Annotated[FlyteDirectory, Qwen14BChat]:
    """Download the Qwen-14B-Chat checkpoint and return it as a FlyteDirectory.

    The FlyteDirectory is returned **annotated** with the Qwen14BChat artifact handle so
    that Union uploads it and versions the artifact automatically.
    """
    from huggingface_hub import snapshot_download
    import union

    workdir = union.current_context().working_directory
    model_dir = os.path.join(workdir, "Qwen-14B-Chat")

    # Hugging Face download (includes git-lfs under the hood).
    snapshot_download(
        repo_id="Qwen/Qwen-14B-Chat",
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # copy real files so they are included in the artifact blob
        resume_download=True,
    )

    return FlyteDirectory(path=model_dir)

# -------------------------------------------------------------------------------------------------
# Workflow ----------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
@union.workflow
def build_qwen14b_artifact() -> Qwen14BChat:  # returns the artifact handle (latest version)
    """Execute once to create (or refresh) the *qwen-14b-chat* artifact."""
    return fetch_model() 