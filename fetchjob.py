#!/usr/bin/env python3
"""Fetch files matching *<filename>* from the HPC cluster."""
import sys, os
from pathlib import Path
import paramiko
from dotenv import dotenv_values

REMOTE_HOST = "hpc.itu.dk"
REMOTE_USER = "jete"
REMOTE_DIRS = ["logs", "RAGNAR/results"]  # relative to home
DEST = Path("fetched")

def main():
    args = sys.argv[1:]
    delete = "-d" in args
    if delete:
        args.remove("-d")

    if not args:
        print("Usage: bash fetchjob.sh [-d] <filename>", file=sys.stderr)
        sys.exit(2)

    filename = args[0]
    env = dotenv_values(Path(__file__).parent / ".env")
    password = env["REMOTE_PASSWORD"]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=password)
    sftp = ssh.open_sftp()

    action = "Deleting" if delete else "Fetching"
    print(f"{action} '*{filename}*' on {REMOTE_USER}@{REMOTE_HOST}")

    count = 0
    for remote_dir in REMOTE_DIRS:
        try:
            entries = sftp.listdir(remote_dir)
        except FileNotFoundError:
            continue
        matches = [e for e in entries if filename in e]
        for name in matches:
            remote_path = f"{remote_dir}/{name}"
            print(f"  {remote_path}")
            if delete:
                sftp.remove(remote_path)
            else:
                DEST.mkdir(exist_ok=True)
                sftp.get(remote_path, str(DEST / name))
            count += 1

    sftp.close()
    ssh.close()
    verb = "deleted" if delete else "fetched"
    dest_note = "" if delete else f" in {DEST}/"
    print(f"Done — {count} file(s) {verb}{dest_note}")

if __name__ == "__main__":
    main()
