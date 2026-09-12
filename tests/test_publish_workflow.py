"""Behavior checks for the manual publishing safeguards without any upload."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

CHECK_SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "check_publish.py"


def _check_publish(tmp_path, confirmation, release="0.1.0a1", selected_ref="v0.1.x"):
    metadata = tmp_path / "fatqat-0.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: fatqat\nVersion: {release}\n", encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "CONFIRM_REF": confirmation,
            "RELEASE_REF": selected_ref,
            "PYTHONPATH": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )


def test_empty_confirmation_keeps_development_builds_unpublished(tmp_path):
    result = _check_publish(tmp_path, "", release="0.1.0.dev1")

    assert result.returncode == 0, result.stderr
    assert "Build only" in result.stdout


@pytest.mark.parametrize(
    "confirmation", ["main", "v0.1.*", "v0.1.x ", "V0.1.x", "v0.1.x\n"]
)
def test_publishing_requires_exact_confirmation(tmp_path, confirmation):
    result = _check_publish(tmp_path, confirmation)

    assert result.returncode != 0
    assert "Confirmation must exactly match" in result.stderr


@pytest.mark.parametrize(
    "selected_ref, release",
    [
        ("v0.1.x", "0.1.0a1"),
        ("main", "0.1.0b1"),
        ("release/0.1", "0.1.0rc1"),
        ("v0.1.0", "0.1.0"),
    ],
)
def test_matching_confirmation_allows_release_versions(tmp_path, selected_ref, release):
    result = _check_publish(tmp_path, selected_ref, release, selected_ref)

    assert result.returncode == 0, result.stderr
    assert f"Confirmed publishing fatqat {release}" in result.stdout


@pytest.mark.parametrize(
    "release", ["0.1.0dev", "0.1.0.dev0", "0.1.0.dev1", "0.1.0a1.dev1", "0.1.0+local"]
)
def test_publishing_rejects_development_and_local_versions(tmp_path, release):
    result = _check_publish(tmp_path, "v0.1.x", release)

    assert result.returncode != 0
    assert "Development and local versions cannot be published" in result.stderr
