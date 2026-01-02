"""
Atomic file I/O operations

All file writes use temp + os.replace pattern for crash safety.
Cross-platform (POSIX + Windows).
"""

from .atomic import (
    write_file_atomic,
    write_json_atomic,
    write_checkpoint_atomic,
    write_torch_artifact_atomic,
    compute_file_sha256,
    get_file_size,
)
from .manifests import (
    ArtifactInfo,
    StepManifest,
    create_step_manifest,
)

__all__ = [
    "write_file_atomic",
    "write_json_atomic",
    "write_checkpoint_atomic",
    "write_torch_artifact_atomic",
    "compute_file_sha256",
    "get_file_size",
    "ArtifactInfo",
    "StepManifest",
    "create_step_manifest",
]
