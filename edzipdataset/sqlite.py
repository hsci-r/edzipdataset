import os
from typing import Optional
from zipfile import ZipFile
import click
from edzip.sqlite import create_sqlite_directory_from_zip

from edzipdataset import derive_sqlite_file_path, derive_sqlite_url_from_zip_url, get_s3_credentials, _get_fs

@click.command()
@click.option("--key", required=False)
@click.option("--secret", required=False, help="AWS secret access key or file from which to read credentials")
@click.option("--endpoint-url", required=False)
@click.argument("s3-url")
@click.argument("sqlite-filename", required=False)
def main(s3_url:str, sqlite_filename:Optional[str] = None, endpoint_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None):
    if secret is not None and os.path.exists(secret):
        credentials = get_s3_credentials(secret)
    else:
        credentials = dict(aws_access_key_id=key, aws_secret_access_key=secret, endpoint_url=endpoint_url)
    
    if sqlite_filename is None:
        sqlite_filename = derive_sqlite_file_path(derive_sqlite_url_from_zip_url(s3_url) ,".")
    with _get_fs(s3_url, credentials).open(s3_url) as zf: # type: ignore
        create_sqlite_directory_from_zip(ZipFile(zf), sqlite_filename) # type: ignore


if __name__ == "__main__":
    main()