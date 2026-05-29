# -*- coding: utf-8 -*-
"""
Download CholecTrack20 from Synapse.

Set credentials in the repo-root ``.env`` file (see ``data/.env.example``)
or export them in your shell. Do NOT hard-code tokens in this file.

    SYNAPSE_EMAIL=you@stanford.edu
    SYNAPSE_AUTH_TOKEN=...           # synapse.org → Settings → Personal Access Tokens
    CHOLECTRACK20_ACCESS_KEY=...       # CAMMA request form
    CHOLECTRACK20_DOWNLOAD_DIR=data/cholectrack20

Then run from the repo root:

    python data/get_dataset.py
"""

from __future__ import annotations

import os
import sys

import requests

# Repo root = parent of data/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_dotenv(path: str | None = None, override: bool = False) -> bool:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Returns True if the file was found and read. Existing environment variables
    are kept unless ``override=True``.
    """
    path = path or os.path.join(_REPO_ROOT, ".env")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value
    return True


def main() -> None:
    env_loaded = load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    if not env_loaded:
        env_loaded = load_dotenv(os.path.join(_REPO_ROOT, "data", ".env"))
    if env_loaded:
        print("Loaded credentials from .env")
    email = os.environ.get("SYNAPSE_EMAIL")
    auth_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    access_key = os.environ.get("CHOLECTRACK20_ACCESS_KEY")
    local_folder = os.environ.get("CHOLECTRACK20_DOWNLOAD_DIR", "data/cholectrack20")

    missing = [
        name
        for name, val in [
            ("SYNAPSE_EMAIL", email),
            ("SYNAPSE_AUTH_TOKEN", auth_token),
            ("CHOLECTRACK20_ACCESS_KEY", access_key),
        ]
        if not val
    ]
    if missing:
        print(
            "Missing environment variables: "
            + ", ".join(missing)
            + f"\nCreate {_REPO_ROOT}/.env from data/.env.example "
            "(or export variables in your shell).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import synapseclient
        import synapseutils
    except ImportError:
        print(
            "Install Synapse client first:\n  pip install synapseclient requests\n"
            "(synapseutils is included inside synapseclient — no separate install)",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Authenticating with Synapse ...")
    syn = synapseclient.login(email=email, authToken=auth_token)

    print("Validating CholecTrack20 access key ...")
    api_url = "https://synapse-response.onrender.com/validate_access"
    user_id = syn.getUserProfile()["ownerId"]
    response = requests.post(
        api_url, json={"access_key": access_key, "synapse_id": user_id}, timeout=60
    )
    if response.status_code != 200:
        print("Failed to validate access key:", response.text, file=sys.stderr)
        sys.exit(1)

    entity_id = response.json()["entity_id"]
    os.makedirs(local_folder, exist_ok=True)

    print(f"Downloading CholecTrack20 to {local_folder} ...")
    synapseutils.syncFromSynapse(syn, entity=entity_id, path=local_folder)
    print("Done.")
    print(
        "\nNext steps:\n"
        "  python scripts/prepare_metadata.py cholectrack20 \\\n"
        f"    --root_dir {local_folder} \\\n"
        "    --output_csv data/cholectrack20_metadata.csv"
    )


if __name__ == "__main__":
    main()
