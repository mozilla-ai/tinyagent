"""Smoke-test that cookbook notebooks execute end-to-end.

Skipped by default unless the relevant API keys are present in the environment.
"""

import os
import pathlib
import subprocess

import pytest

REQUIRED_ENV = ("MISTRAL_API_KEY",)


def _missing_keys() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


pytestmark = pytest.mark.skipif(
    bool(_missing_keys()),
    reason=f"Missing env keys for cookbook execution: {_missing_keys()}",
)


@pytest.mark.parametrize(
    "notebook_path",
    list(pathlib.Path("docs/cookbook").glob("*.ipynb")),
    ids=lambda x: x.stem,
)
@pytest.mark.timeout(180)
def test_cookbook_notebook(
    notebook_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute a cookbook notebook and surface failures."""
    env = {
        k: os.environ[k]
        for k in os.environ
        if k.endswith("_API_KEY") or k in ("HF_TOKEN", "HF_ENDPOINT", "PATH")
    }
    env["IN_PYTEST"] = "1"

    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "jupyter",
                "execute",
                notebook_path.name,
                "--allow-errors",
                "--output",
                f"executed_{notebook_path.name}",
            ],
            cwd="docs/cookbook",
            env=env,
            timeout=170,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if e.stdout else "(no stdout captured)"
        stderr = e.stderr.decode() if e.stderr else "(no stderr captured)"
        pytest.fail(
            f"Notebook {notebook_path.name} timed out after ~3 minutes\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

    if result.returncode != 0:
        stdout = result.stdout.decode() if result.stdout else "(no stdout captured)"
        stderr = result.stderr.decode() if result.stderr else "(no stderr captured)"
        pytest.fail(
            f"Notebook {notebook_path.name} failed with return code {result.returncode}\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )
