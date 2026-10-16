# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Centralized log and results management.

This module provides a LogManager class that handles all path generation,
directory creation, and file I/O for benchmark results and logs.
"""

import json
import logging
import os
import socket
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:                                    # POSIX; without it no run directory is shared (see below)
    import fcntl
except ImportError:                     # pragma: no cover - gbench runs on Linux
    fcntl = None

logger = logging.getLogger(__name__)

#: The file in a run directory whose exclusive lock (flock) the run writing there holds for as
#: long as it runs. A second run that finds it held does not share the directory.
RUN_LOCK_NAME = ".gbench-run.lock"

#: How many suffixed names (<name>-2, <name>-3, ...) create_unique_dir tries.
UNIQUE_DIR_ATTEMPTS = 1000


def create_unique_dir(parent: Path | str, name: str,
                      attempts: int = UNIQUE_DIR_ATTEMPTS) -> Path:
    """Create a directory of its own under `parent`, atomically: <name>, else <name>-2,
    <name>-3, ... mkdir without exist_ok is the atomic claim: exactly one process gets each name,
    so runs (or merges) started in the same second on one parent never share a directory."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, attempts + 1):
        candidate = parent / (name if n == 1 else f"{name}-{n}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a directory for {name} under {parent} "
                       f"({attempts} names taken)")


class _RunLock:
    """One run's exclusive flock on a run directory's RUN_LOCK_NAME file (a descriptor of its
    own: a second LogManager, in this process or another, is a second run and conflicts). The
    descriptor is closed (and the lock released) when the LogManager holding it is
    garbage-collected or released, or by the OS when the process ends, however it ends: a
    crashed run leaves no stale lock."""

    def __init__(self, run_dir: Path, fd: int):
        self.run_dir = run_dir
        self.fd = fd
        self._finalizer = weakref.finalize(self, os.close, fd)

    def release(self) -> None:
        self._finalizer()


def _lock_holder_text(path: Path) -> str:
    """Who holds `path` (what the holder wrote into it), for the message."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        text = ""
    return text or "a run that did not record itself"


def lock_run_dir(run_dir: Path) -> Tuple[Optional[_RunLock], str]:
    """Take a run's exclusive lock on `run_dir` (its RUN_LOCK_NAME file, created when missing),
    without waiting. Returns (lock, "") when taken, (None, "in use by <holder>") when another
    run holds it (another LogManager, of this process or any other), and (None, "cannot lock:
    <why>") when the filesystem cannot lock files (no one can then tell whether a run is still
    writing there)."""
    if fcntl is None:
        return None, "cannot lock: this platform has no fcntl.flock"
    path = run_dir / RUN_LOCK_NAME
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        return None, f"cannot lock: {path} cannot be opened ({e})"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, f"in use by {_lock_holder_text(path)}"
    except OSError as e:
        os.close(fd)
        return None, f"cannot lock: {path} ({e})"
    lock = _RunLock(run_dir, fd)
    try:
        os.ftruncate(fd, 0)
        # the short host name only (the node, not its domain)
        host = socket.gethostname().split(".", 1)[0] or "an unnamed host"
        os.write(fd, (f"pid {os.getpid()} on {host}, started "
                      f"{datetime.now().isoformat(timespec='seconds')}\n").encode("utf-8"))
    except OSError:
        pass                            # the lock holds; only the holder's note is missing
    return lock, ""


def run_dir_in_use(run_dir: Path) -> str:
    """Why a run could not write into `run_dir` ("in use by <holder>", or "cannot lock: ..."),
    or "" when it could. Creates nothing (for a dry run): a directory without a lock file was
    never locked, and a lock file that cannot be probed read-only (some network filesystems)
    reads as free."""
    if fcntl is None:
        return "cannot lock: this platform has no fcntl.flock"
    path = run_dir / RUN_LOCK_NAME
    if not path.is_file():
        return ""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return ""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return ""
    except BlockingIOError:
        return f"in use by {_lock_holder_text(path)}"
    except OSError:
        return ""
    finally:
        os.close(fd)


def _json_default(o: Any) -> Any:
    """json.dump fallback for values the stdlib encoder cannot handle.

    Result dicts routinely carry numpy scalars - e.g. a stress level's
    ``passed`` is ``numpy.bool_`` (a numpy-float TTFT compared to the
    threshold) and stats are ``numpy.float64``. numpy 2.x names its bool
    scalar ``bool``, so json raises "Object of type bool is not JSON
    serializable". numpy scalars expose ``.item()``, which returns the
    native Python equivalent (bool/int/float) so numbers stay numbers;
    anything else falls back to ``str`` rather than crashing the write.
    """
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(o)


def redact_final_text(text: str) -> str:
    """`text` with every credential value replaced (gbench.utils.redaction.redact_text):
    applied to the FINAL text a writer produces, after serialization, so a value json, csv or
    str() turned into text (a set, an exception, a path, bytes, any object) cannot carry a key
    past it. Imported on use, so importing LogManager does not load the eval suites."""
    from .redaction import redact_text
    return redact_text(text)


def write_redacted_text(path: Path | str, text: Any, *, encoding: str = "utf-8",
                        newline: Optional[str] = None) -> Path:
    """The one write path for text files under the results directory: `text` (bytes are
    decoded as UTF-8, anything else is str()'d) is redacted and written to `path`."""
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", errors="replace")
    elif not isinstance(text, str):
        text = "" if text is None else str(text)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline=newline) as f:
        f.write(redact_final_text(text))
    return path


#: How much serialized JSON write_redacted_json buffers before it redacts and writes a piece.
_JSON_STREAM_CHUNK = 8 << 20


def write_redacted_json(path: Path | str, data: Any, *, indent: Optional[int] = 2,
                        default: Any = None, **dump_kwargs: Any) -> Path:
    """Serialize `data` as JSON (`default` falls back to _json_default), redact the text and
    write it: the JSON writer every result, summary and metadata file goes through. The text is
    encoded and redacted in pieces of about _JSON_STREAM_CHUNK characters, never as one string
    (a result's traces can run to hundreds of MB): each piece is redacted with the end of the one
    before it held back, a credential value's length less one, so a value split between two
    pieces is still found."""
    from .redaction import _redaction_needles, redact_text, secret_values
    needles = _redaction_needles(tuple(secret_values()))
    keep = max((len(n) for n in needles), default=1) - 1
    encoder = json.JSONEncoder(indent=indent, default=default or _json_default, **dump_kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        pending: list = []
        size = 0
        tail = ""
        for piece in encoder.iterencode(data):
            pending.append(piece)
            size += len(piece)
            if size < _JSON_STREAM_CHUNK:
                continue
            text = redact_text(tail + "".join(pending))
            pending, size = [], 0
            if keep > 0:
                tail = text[-keep:] if len(text) > keep else text
                text = text[:-keep] if len(text) > keep else ""
            else:
                tail = ""
            f.write(text)
        f.write(redact_text(tail + "".join(pending)))
    return path


def write_redacted_csv(path: Path | str, rows: list, fieldnames: list) -> Path:
    """Render `rows` as CSV in memory, redact the text and write it (the rows' strings are
    redacted first too, since CSV quoting can change how a value is spelled in the text)."""
    import csv
    import io
    from .redaction import redact_secrets
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(redact_secrets(rows))
    return write_redacted_text(path, buf.getvalue(), newline="")


def foreign_run_reason(run_dir: Path, identity: Optional[Dict[str, Any]]) -> str:
    """Why `run_dir` is another run's directory, or "" when this run (`identity`: {"host",
    "shard", "gbench"}) may continue it: its metadata.json records a different host, shard or
    gbench commit. Only what both record is compared (an interrupted run may not have written
    "gbench" yet); no metadata.json or no identity compares nothing."""
    if not identity:
        return ""
    try:
        meta = json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(meta, dict):
        return ""
    diffs = []
    for key in ("host", "shard"):
        if key in meta and key in identity and meta.get(key) != identity.get(key):
            diffs.append(f"{key} {meta.get(key)!r}, this run {identity.get(key)!r}")
    theirs = (meta.get("gbench") or {}).get("commit") if isinstance(meta.get("gbench"), dict) \
        else None
    mine = (identity.get("gbench") or {}).get("commit") \
        if isinstance(identity.get("gbench"), dict) else None
    if theirs and mine and theirs != mine:
        diffs.append(f"gbench commit {theirs}, this run {mine}")
    if not diffs:
        return ""
    return "a run directory of another node, shard or commit (" + "; ".join(diffs) + ")"


class LogManager:
    """Centralized manager for benchmark results and logs.
    
    Handles:
    - Timestamped results directory creation
    - Path generation for results and logs
    - File I/O for saving/loading results
    - Directory structure management
    """
    
    def __init__(self, base_dir: Path | str = "./results", *, create_dirs: bool = True,
                 identity: Any = None):
        """Initialize LogManager with timestamped results directory.

        Args:
            base_dir: Base directory for results (default: ./results)
            create_dirs: False for a run that writes nothing (a dry run): the run directory
                is resolved (base_dir itself when it already is a run directory, else
                base_dir/<timestamp>, the name a real run claims first) but nothing is created
                on disk and GBENCH_RESULTS_DIR is not exported.

        The run directory is this run's alone. base_dir/<timestamp> is claimed atomically
        (create_unique_dir). base_dir itself is reused when it already is a run directory
        (metadata.json or an eval_*.json in it: an interrupted run continued, --skip-existing),
        but only by one run at a time: the run writing there holds the exclusive lock on its
        RUN_LOCK_NAME file for as long as it runs. A run that finds that lock held by another run
        (swarm nodes pointed at one shared --results-dir), or on a filesystem that cannot lock,
        writes its own base_dir/<timestamp> instead and says so (`shared_dir_note`), so it
        never overwrites another run's metadata.json, summary.json, results or logs. The same
        holds for a run directory another node, shard or gbench commit wrote, even after that run
        ended: `identity` ({"host", "shard", "gbench"}, as metadata.json records them, or a
        callable returning it, called only when base_dir is a run directory) is compared with the
        directory's metadata.json (foreign_run_reason), so --merge's commit check never reads
        another run's metadata for the files there. --skip-existing still searches all of
        base_dir.
        """
        self.base_dir = Path(base_dir)
        self.creates_dirs = bool(create_dirs)
        #: why base_dir, an existing run directory, was not reused (None when it was, or when
        #: base_dir was not a run directory)
        self.shared_dir_note: Optional[str] = None
        self._run_lock: Optional[_RunLock] = None
        stamp = datetime.now().strftime("%m-%d-%H%M%S")

        # If base_dir is already an existing run directory (contains metadata.json or
        # eval_*.json), reuse it directly - unless another run is writing there.
        if self.base_dir.exists() and (
            (self.base_dir / "metadata.json").exists()
            or any(self.base_dir.glob("eval_*.json"))
        ):
            if self.creates_dirs:
                self._run_lock, why = lock_run_dir(self.base_dir)
            else:
                why = run_dir_in_use(self.base_dir)
            if not why:
                why = foreign_run_reason(self.base_dir,
                                         identity() if callable(identity) else identity)
                if why and self._run_lock is not None:
                    self._run_lock.release()
                    self._run_lock = None
            if not why:
                self.results_dir = self.base_dir
                self.timestamp = self.base_dir.name
            else:
                self.shared_dir_note = why
                if self.creates_dirs:
                    self.results_dir = self._create_run_dir(stamp)
                    self._run_lock, _ = lock_run_dir(self.results_dir)
                else:
                    self.results_dir = self.base_dir / stamp
                self.timestamp = self.results_dir.name
                logger.warning(
                    f"--results-dir {self.base_dir} is a run directory this run does not share "
                    f"({why}); this run writes its own directory {self.results_dir} instead, so "
                    "neither run overwrites the other's results (--skip-existing still searches "
                    f"all of {self.base_dir}).")
        elif not self.creates_dirs:
            self.timestamp = stamp
            self.results_dir = self.base_dir / self.timestamp
        else:
            self.results_dir = self._create_run_dir(stamp)
            self.timestamp = self.results_dir.name
            # A fresh directory is this run's by the atomic claim; the lock tells a later run
            # pointed at it (once it holds metadata.json) that this one is still writing.
            self._run_lock, _ = lock_run_dir(self.results_dir)

        if not self.creates_dirs:
            logger.info(f"No results directory is created under {self.base_dir} (this run "
                        "writes nothing)")
            return

        # Create directory structure
        self._setup_directories()

        # Expose the resolved run directory so eval runners that write their own auxiliary
        # artifacts (e.g. tau2's per-task simulation traces) can co-locate them with the
        # eval JSONs without threading the path through every eval signature.
        os.environ["GBENCH_RESULTS_DIR"] = str(self.results_dir)

        logger.info(f"Results will be saved to: {self.results_dir}")
    
    #: How many suffixed names (<stamp>-2, <stamp>-3, ...) _create_run_dir tries.
    RUN_DIR_ATTEMPTS = UNIQUE_DIR_ATTEMPTS

    def _create_run_dir(self, stamp: str) -> Path:
        """Create this run's own directory under base_dir, atomically: <stamp>, else <stamp>-2,
        <stamp>-3, ... (create_unique_dir). The stamp has one-second resolution, so runs started
        in the same second on one --results-dir (parallel jobs, shard nodes on a shared disk)
        would otherwise share a directory and overwrite each other's metadata.json, summary and
        results."""
        return create_unique_dir(self.base_dir, stamp, self.RUN_DIR_ATTEMPTS)

    def release_run_lock(self) -> None:
        """Release this run's lock on its run directory (it is released at exit anyway)."""
        if self._run_lock is not None:
            lock, self._run_lock = self._run_lock, None
            lock.release()

    def _setup_directories(self):
        """Create the directory structure for results."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "performance").mkdir(exist_ok=True)
        (self.results_dir / "quality").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
    
    # === Path Generation ===
    
    def get_serving_result_path(
        self,
        model_name: str,
        format: str,
        batch_size: int,
        multimodal: bool = False,
        campaign: Optional[str] = None,
    ) -> Path:
        """Generate path for serving benchmark result file (one per campaign)."""
        suffix = "_mm" if multimodal else ""
        camp = f"_{campaign}" if isinstance(campaign, str) and campaign else ""
        filename = f"serve_{model_name}_{format}{camp}_b{batch_size}{suffix}.json"
        return self.results_dir / "performance" / filename
    
    def get_serving_log_path(
        self,
        model_name: str,
        format: str,
        batch_size: int,
        multimodal: bool = False,
    ) -> Path:
        """Generate path for serving benchmark log file."""
        suffix = "_mm" if multimodal else ""
        filename = f"serve_{model_name}_{format}_b{batch_size}{suffix}.log"
        return self.results_dir / "logs" / filename

    def get_server_log_path(self, model_name: str, format: str) -> Path:
        """Generate path for vLLM server stdout/stderr logs.
        
        Unlike benchmark logs, this is for the server process itself,
        which is shared across multiple benchmark configurations.
        """
        filename = f"server_{model_name}_{format}.log"
        return self.results_dir / "logs" / filename
    
    def get_throughput_result_path(
        self,
        model_name: str,
        format: str,
        input_length: int,
        output_length: int,
        batch_size: int,
    ) -> Path:
        """Generate path for throughput benchmark result file."""
        filename = (
            f"throughput_{model_name}_{format}_"
            f"in{input_length}_out{output_length}_b{batch_size}.json"
        )
        return self.results_dir / "performance" / filename
    
    def get_throughput_log_path(
        self,
        model_name: str,
        format: str,
        input_length: int,
        output_length: int,
        batch_size: int,
    ) -> Path:
        """Generate path for throughput benchmark log file."""
        filename = (
            f"throughput_{model_name}_{format}_"
            f"in{input_length}_out{output_length}_b{batch_size}.log"
        )
        return self.results_dir / "logs" / filename
    
    def get_stress_result_path(
        self,
        model_name: str,
        format: str,
        mode: str,
    ) -> Path:
        """Generate path for stress test result file.
        
        Args:
            model_name: Model short name
            format: Model format (hf, or remote-endpoint)
            mode: Test mode ('text' or 'mm')
        """
        filename = f"stress_{model_name}_{format}_{mode}.json"
        return self.results_dir / "performance" / filename
    
    def get_quality_result_path(
        self,
        model_name: str,
        format: str,
    ) -> Path:
        """Generate path for quality benchmark result file."""
        filename = f"quality_{model_name}_{format}.json"
        return self.results_dir / "quality" / filename

    def get_quality_log_path(
        self,
        model_name: str,
        format: str,
    ) -> Path:
        """Generate path for quality benchmark log file."""
        filename = f"quality_{model_name}_{format}.log"
        return self.results_dir / "logs" / filename
    
    # === File I/O ===
    
    def save_result(self, path: Path, data: dict) -> None:
        """Save benchmark result to JSON file.
        
        Args:
            path: Path to save file
            data: Result data dictionary
        """
        write_redacted_json(path, data)
        logger.debug(f"Saved result to: {path}")

    def save_metadata(self, metadata: dict) -> None:
        """Save run metadata to metadata.json."""
        path = self.results_dir / "metadata.json"
        write_redacted_json(path, metadata)
        logger.debug(f"Saved run metadata to: {path}")

    def save_summary(self, summary_data: dict) -> Path:
        """Save aggregated run summary to summary.json."""
        path = self.results_dir / "summary.json"
        write_redacted_json(path, summary_data)
        logger.debug(f"Saved aggregated summary to: {path}")
        return path

    def save_csv(self, filename: str, rows: list[dict], fieldnames: list[str]) -> Path:
        """Save rows to CSV file in results directory."""
        path = self.results_dir / filename
        write_redacted_csv(path, rows, fieldnames)
        logger.debug(f"Saved CSV report to: {path}")
        return path
    
    def load_result(self, path: Path) -> Optional[dict]:
        """Load benchmark result from JSON file.
        
        Args:
            path: Path to result file
            
        Returns:
            Result data dictionary, or None if file doesn't exist
        """
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    
    def save_log(self, path: Path, stdout: str, stderr: str) -> None:
        """Save benchmark log to file.
        
        Args:
            path: Path to log file
            stdout: Standard output content
            stderr: Standard error content
        """
        def _text(value: Any) -> str:
            if isinstance(value, (bytes, bytearray)):
                return bytes(value).decode("utf-8", errors="replace")
            return "" if value is None else str(value)
        write_redacted_text(path, f"=== STDOUT ===\n{_text(stdout)}\n"
                                  f"=== STDERR ===\n{_text(stderr)}\n")
        logger.debug(f"Saved log to: {path}")
    
    def result_exists(self, path: Path) -> bool:
        """Check if a result file already exists."""
        return path.exists()
    
    # === Summary Methods ===
    
    def list_result_files(self) -> list[Path]:
        """List all result files in the performance directory."""
        perf_dir = self.results_dir / "performance"
        if not perf_dir.exists():
            return []
        return sorted(perf_dir.glob("*.json"))
    
    def list_quality_result_files(self) -> list[Path]:
        """List all result files in the quality directory."""
        qual_dir = self.results_dir / "quality"
        if not qual_dir.exists():
            return []
        return sorted(qual_dir.glob("*.json"))
    
    def list_log_files(self) -> list[Path]:
        """List all log files in the logs directory."""
        logs_dir = self.results_dir / "logs"
        if not logs_dir.exists():
            return []
        return sorted(logs_dir.glob("*.log"))
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all results in this run.
        
        Returns:
            Dictionary with result counts and paths
        """
        perf_files = self.list_result_files()
        qual_files = self.list_quality_result_files()
        result_files = perf_files + qual_files
        log_files = self.list_log_files()
        
        return {
            "results_dir": str(self.results_dir.absolute()),
            "timestamp": self.timestamp,
            "result_count": len(result_files),
            "log_count": len(log_files),
            "result_files": [f.name for f in result_files],
            "log_files": [f.name for f in log_files],
        }
