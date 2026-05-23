"""
GCS write-back utility.

When the environment variable GCS_BUCKET is set, every checkpoint
and result file is uploaded to GCS immediately after the local save.
When GCS_BUCKET is not set, all methods are silent no-ops — the code
runs identically on a local machine or a VM without a bucket configured.

Environment variables
---------------------
  GCS_BUCKET   — bucket name, e.g. "vks_bucket"          (required to enable)
  GCS_PREFIX   — path prefix inside bucket, e.g. "MS_PROJECT"  (default: "MS_PROJECT")

Usage
-----
  from ids_ips.utils.gcs import gcs
  gcs.upload(local_path)
  gcs.upload(local_path, 'checkpoints/x.pt')
  gcs.sync_dir(local_dir, 'results')
  gcs.download('checkpoints/mamba_final.pt', local_path)
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class GCSManager:
    """
    Thin wrapper around google-cloud-storage.

    All public methods are safe to call even when GCS is disabled —
    they simply return immediately without raising any error.
    """

    def __init__(self):
        self.bucket_name: str        = os.environ.get("GCS_BUCKET", "").strip()
        self.prefix:      str        = os.environ.get("GCS_PREFIX", "MS_PROJECT").strip("/")
        self.enabled:     bool       = bool(self.bucket_name)
        self._client                 = None
        self._bucket                 = None

        if self.enabled:
            logger.info(
                "GCS write-back enabled: gs://%s/%s",
                self.bucket_name, self.prefix
            )
        else:
            logger.debug(
                "GCS_BUCKET not set — write-back disabled. "
                "Set GCS_BUCKET=<your-bucket> to enable."
            )

    def _get_bucket(self):
        """Lazy-initialise the GCS client (avoids import cost when disabled)."""
        if self._bucket is None:
            try:
                from google.cloud import storage as gcs_lib
                self._client = gcs_lib.Client()
                self._bucket = self._client.bucket(self.bucket_name)
                logger.info("GCS client initialised for bucket: %s", self.bucket_name)
            except ImportError:
                logger.error(
                    "google-cloud-storage not installed. "
                    "Run: pip install google-cloud-storage"
                )
                self.enabled = False
            except Exception as e:
                logger.error("Failed to initialise GCS client: %s", e)
                self.enabled = False
        return self._bucket

    def _blob_name(self, subpath: str) -> str:
        """Combine prefix + subpath into a GCS blob name."""
        return f"{self.prefix}/{subpath.lstrip('/')}"

    def upload(self, local_path, gcs_subpath: Optional[str] = None) -> bool:
        """
        Upload a local file to GCS.

        Parameters
        ----------
        local_path   : local file to upload (str or Path)
        gcs_subpath  : destination inside prefix, e.g. "checkpoints/mamba.pt"
                       Defaults to the file's name (uploaded directly under prefix/).

        Returns True on success, False on failure / disabled.
        """
        if not self.enabled:
            return False
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("GCS upload skipped — file not found: %s", local_path)
            return False

        bucket = self._get_bucket()
        if bucket is None:
            return False

        subpath   = gcs_subpath or local_path.name
        blob_name = self._blob_name(subpath)
        try:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            logger.info("GCS   gs://%s/%s", self.bucket_name, blob_name)
            return True
        except Exception as e:
            logger.error("GCS upload failed (%s  %s): %s", local_path, blob_name, e)
            return False

    def download(self, gcs_subpath: str, local_path) -> bool:
        """
        Download a file from GCS to local disk.

        Parameters
        ----------
        gcs_subpath : source inside prefix, e.g. "checkpoints/mamba_final.pt"
        local_path  : local destination (str or Path)
        """
        if not self.enabled:
            return False

        bucket = self._get_bucket()
        if bucket is None:
            return False

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_name = self._blob_name(gcs_subpath)
        try:
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
            logger.info("GCS   gs://%s/%s  %s", self.bucket_name, blob_name, local_path)
            return True
        except Exception as e:
            logger.error("GCS download failed (%s  %s): %s", blob_name, local_path, e)
            return False

    def sync_dir(self, local_dir, gcs_dir: Optional[str] = None) -> int:
        """
        Upload every file in local_dir to GCS.

        Parameters
        ----------
        local_dir  : local directory to sync
        gcs_dir    : destination sub-directory inside prefix
                     Defaults to the directory's name.

        Returns the number of successfully uploaded files.
        """
        if not self.enabled:
            return 0

        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            logger.warning("GCS sync skipped — not a directory: %s", local_dir)
            return 0

        gcs_dir  = gcs_dir or local_dir.name
        uploaded = 0
        for f in sorted(local_dir.iterdir()):
            if f.is_file():
                ok = self.upload(f, f"{gcs_dir}/{f.name}")
                uploaded += int(ok)
        logger.info("GCS sync complete: %d files  gs://%s/%s/%s",
                    uploaded, self.bucket_name, self.prefix, gcs_dir)
        return uploaded

    def status(self) -> str:
        if self.enabled:
            return f"ENABLED    gs://{self.bucket_name}/{self.prefix}"
        return "DISABLED (set GCS_BUCKET env var to enable)"

gcs = GCSManager()
