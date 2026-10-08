"""Shared settings for workflow/vendored/*.smk: config file, repo paths, job prefixes."""
from pathlib import Path


configfile: str(Path(workflow.basedir) / "config.yaml")


REPO_DIR = Path(workflow.basedir).parents[1]
PYTHON_BIN = REPO_DIR / ".pixi/envs/default/bin/python"
SCRIPTS = REPO_DIR / "scripts/vendored"


def cfg_path(key):
    value = str(config[key])
    if value.startswith("<"):
        raise ValueError(f"set '{key}' in workflow/vendored/config.yaml or via --configfile (currently {value})")
    return Path(value)


def job_prefix(kind):
    value = config[f"slurm_{kind}"]
    return f"bash {REPO_DIR}/workflow/vendored/run_local.sh" if value == "local" else value
