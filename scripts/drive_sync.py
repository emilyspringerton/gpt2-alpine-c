#!/usr/bin/env python3
"""
Upload/download training artifacts via IDUNA Drive API.

Authenticates as EMILY-TRAINING agent. Requires:
  IDUNA_BASE_URL      — default: http://localhost:8080
  IDUNA_AGENT_NAME    — default: EMILY-TRAINING
  IDUNA_AGENT_SECRET  — from IDUNA var/agent-secrets.env

Usage:
  # Upload a file
  python3 scripts/drive_sync.py --upload /tmp/emily-corpus.jsonl

  # List files in Drive folder
  python3 scripts/drive_sync.py --list

  # Download matching files
  python3 scripts/drive_sync.py --download --pattern "checkpoint*.tar.gz" --dest ./checkpoints/

  # Download a specific file by ID
  python3 scripts/drive_sync.py --get <file_id>
"""

import argparse
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class IDUNAClient:
    def __init__(self, base_url: str, agent_name: str, agent_secret: str):
        self.base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.agent_secret = agent_secret
        self._token: str | None = None

    def _authenticate(self) -> str:
        payload = json.dumps({
            "agent_name": self.agent_name,
            "agent_secret": self.agent_secret,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/auth/agent",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
                return body["token"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"IDUNA auth failed ({e.code}): {body}") from e

    def token(self) -> str:
        if not self._token:
            self._token = self._authenticate()
        return self._token

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token()}"}

    def upload(self, file_path: Path) -> dict:
        data = file_path.read_bytes()
        filename = file_path.name

        mime_map = {
            ".jsonl": "application/x-ndjson",
            ".json": "application/json",
            ".txt": "text/plain",
            ".py": "text/x-python",
            ".tar.gz": "application/gzip",
            ".bin": "application/octet-stream",
        }
        suffix = "".join(file_path.suffixes[-2:]) if file_path.name.endswith(".tar.gz") else file_path.suffix
        mime = mime_map.get(suffix, "application/octet-stream")

        boundary = b"----EMILYTrainingBoundary"
        body_parts = [
            b"--" + boundary + b"\r\n",
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            data,
            b"\r\n--" + boundary + b"--\r\n",
        ]
        body = b"".join(body_parts)

        headers = self._auth_headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary.decode()}"
        headers["Content-Length"] = str(len(body))

        req = urllib.request.Request(
            f"{self.base_url}/api/v1/drive/upload",
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_err = e.read().decode()
            raise RuntimeError(f"upload failed ({e.code}): {body_err}") from e

    def list_files(self) -> list[dict]:
        headers = self._auth_headers()
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/drive/files",
            method="GET",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
                return result.get("files", [])
        except urllib.error.HTTPError as e:
            body_err = e.read().decode()
            raise RuntimeError(f"list failed ({e.code}): {body_err}") from e

    def get_file_info(self, file_id: str) -> dict:
        headers = self._auth_headers()
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/drive/files/{file_id}",
            method="GET",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_err = e.read().decode()
            raise RuntimeError(f"get_file failed ({e.code}): {body_err}") from e


def cmd_upload(client: IDUNAClient, file_paths: list[str]) -> None:
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        size_kb = path.stat().st_size / 1024
        print(f"Uploading {path.name} ({size_kb:.1f} KB)...")
        result = client.upload(path)
        print(f"  OK: id={result.get('id', '?')} name={result.get('name', '?')} link={result.get('web_view_link', '')}")


def cmd_list(client: IDUNAClient) -> None:
    files = client.list_files()
    if not files:
        print("(no files)")
        return
    print(f"{'ID':<40} {'Name':<40} {'Size':>10}")
    print("-" * 95)
    for f in files:
        size = f.get("size", 0)
        size_str = f"{int(size) / 1024:.1f} KB" if size else "?"
        print(f"{f.get('id','?'):<40} {f.get('name','?'):<40} {size_str:>10}")


def cmd_download(client: IDUNAClient, pattern: str | None, dest: str) -> None:
    files = client.list_files()
    if pattern:
        files = [f for f in files if fnmatch.fnmatch(f.get("name", ""), pattern)]
    if not files:
        print("No matching files found.")
        return

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    for f in files:
        file_id = f.get("id", "")
        name = f.get("name", file_id)
        web_link = f.get("web_view_link", "")
        print(f"  {name}: {web_link}")
        print(f"  (download manually from Drive — IDUNA Drive API provides metadata only)")
        print(f"  File ID: {file_id}")


def cmd_get(client: IDUNAClient, file_id: str) -> None:
    info = client.get_file_info(file_id)
    print(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Sync training artifacts via IDUNA Drive API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--upload", nargs="+", metavar="FILE", help="Upload files to Drive")
    group.add_argument("--list", action="store_true", help="List files in Drive folder")
    group.add_argument("--download", action="store_true", help="Download files (list with links)")
    group.add_argument("--get", metavar="FILE_ID", help="Get file info by ID")

    parser.add_argument("--pattern", default=None, help="Filter pattern for --download (e.g. 'checkpoint*.tar.gz')")
    parser.add_argument("--dest", default="./downloads", help="Destination dir for --download")

    args = parser.parse_args()

    base_url = get_env("IDUNA_BASE_URL", "http://localhost:8080")
    agent_name = get_env("IDUNA_AGENT_NAME", "EMILY-TRAINING")
    agent_secret = get_env("IDUNA_AGENT_SECRET")

    if not agent_secret:
        print("ERROR: IDUNA_AGENT_SECRET not set.", file=sys.stderr)
        print("  Source IDUNA's var/agent-secrets.env or set IDUNA_AGENT_SECRET manually.", file=sys.stderr)
        sys.exit(1)

    client = IDUNAClient(base_url, agent_name, agent_secret)

    try:
        if args.upload:
            cmd_upload(client, args.upload)
        elif args.list:
            cmd_list(client)
        elif args.download:
            cmd_download(client, args.pattern, args.dest)
        elif args.get:
            cmd_get(client, args.get)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
