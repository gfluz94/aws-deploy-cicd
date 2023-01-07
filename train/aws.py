import os
import boto3
from pathlib import Path
from botocore.exceptions import ClientError

from train.exceptions import S3ClientError, InvalidBucketName, MissingCredentials

SUCCESS_STATUS_CODE = "200"


def fetch_s3_data(
    bucket_name: str,
    prefix: str,
    filename: str,
    target_folder: str,
    aws_access_key: str,
    aws_secret_key: str,
) -> None:
    """Function that fetches data from specified S3 bucket and saves it to the target folder.
    Args:
        bucket_name (str): Name of the bucket in S3 where files are nested
        prefix (str): Key name inside bucket.
        filename (str): Filename in Amazon S3.
        target_folder (str): Path to folder where data is going to be saved.
        aws_access_key (str): AWS Access Key in order to fetch data from S3
        aws_secret_key (str): AWS Secret Key in order to fetch data from S3
    Raises:
        S3ClientError: Download from S3 fails - Status Code is different from 200.
    """
    if bucket_name == "" or not isinstance(bucket_name, str):
        raise InvalidBucketName("Please provide a valid string for bucket name!")

    if not aws_access_key or not aws_secret_key:
        raise MissingCredentials("Please provide valid AWS Credentials!")

    client = boto3.client(
        "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )
    try:
        client.download_file(
            Bucket=bucket_name,
            Key=f"{prefix}/{filename}",
            Filename=os.path.join(target_folder, filename),
        )
    except ClientError as exc:
        raise S3ClientError(
            "Please check either bucket name or permissions to access S3 bucket!"
        ) from exc


def dump_s3_data(
    local_filepath: str,
    bucket_name: str,
    prefix: str,
    aws_access_key: str,
    aws_secret_key: str,
) -> None:
    """Function that dumps a file to a specified S3 bucket.
    Args:
        local_filepath (str): Local file that needs to be dumped to S3.
        bucket_name (str): Name of the bucket in S3 where files are nested
        key_name (str): Key name inside bucket.
        aws_access_key (str): AWS Access Key in order to fetch data from S3
        aws_secret_key (str): AWS Secret Key in order to fetch data from S3
    Raises:
        S3ClientError: Download from S3 fails - Status Code is different from 200.
    """
    if bucket_name == "" or not isinstance(bucket_name, str):
        raise InvalidBucketName("Please provide a valid string for bucket name!")

    if not aws_access_key or not aws_secret_key:
        raise MissingCredentials("Please provide valid AWS Credentials!")

    client = boto3.client(
        "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )
    local_filepath = Path(local_filepath)
    try:
        client.upload_file(
            Filename=local_filepath,
            Bucket=bucket_name,
            Key=f"{prefix}/{local_filepath.name}",
        )
    except ClientError as exc:
        raise S3ClientError(
            "Please check either bucket name or permissions to upload data to S3 bucket!"
        ) from exc
