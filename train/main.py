import os
from argparse import ArgumentParser
from dotenv import load_dotenv
import pandas as pd
import logging

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from aws import fetch_s3_data
from exceptions import EnvironmentVariablesMissing

TARGET_FILE = "dataset.csv"


if __name__ == "__main__":
    logger.info("Defining parameters...")
    parser = ArgumentParser(
        description="Input parameters for training a classifier and dumping model to S3."
    )
    parser.add_argument(
        "--data-folder",
        metavar="N",
        type=str,
        help="Local folder where data will be uploaded to.",
        default="data",
    )
    parser.add_argument(
        "--output-folder",
        metavar="N",
        type=str,
        help="Local folder where data will be dumped to.",
        default="data",
    )
    parser.add_argument(
        "--s3-data-bucket",
        metavar="N",
        type=str,
        help="Name of the bucket in AWS S3 where data is currelty stored.",
        default="data-cicd-gfluz94",
    )
    parser.add_argument(
        "--s3-data-prefix",
        metavar="N",
        type=str,
        help="Name of the bucket in AWS S3 where data is currelty stored.",
        default="data",
    )
    parser.add_argument(
        "--aws-access-key-env",
        metavar="N",
        type=str,
        help="Name of the environment variable for AWS Access Key.",
        default="AWS_ACCESS_KEY",
    )
    parser.add_argument(
        "--aws-secret-key-env",
        metavar="N",
        type=str,
        help="Name of the environment variable for AWS Secret Key.",
        default="AWS_SECRET_KEY",
    )
    args = parser.parse_args()
    load_dotenv()
    logger.info("Parameters defined!")

    logger.info("Finding and reading raw data...")
    DATA_FOLDER = os.path.join(os.curdir, args.data_folder)
    FILES_IN_FOLDER = os.listdir(DATA_FOLDER)
    if TARGET_FILE not in FILES_IN_FOLDER:
        logger.info(
            "Data not found in %s. Downloading from `%s` S3 bucket...",
            DATA_FOLDER,
            args.s3_data_bucket,
        )
        aws_access_key = os.getenv(key=args.aws_access_key_env)
        aws_secret_key = os.getenv(key=args.aws_secret_key_env)
        if not aws_access_key or not aws_secret_key:
            raise EnvironmentVariablesMissing("AWS Credentials not set accordingly.`")
        fetch_s3_data(
            bucket_name=args.s3_data_bucket,
            prefix=args.s3_data_prefix,
            filename=TARGET_FILE,
            target_folder=DATA_FOLDER,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
        )
        logger.info("Dataset successfully uploaded to local folder %s...", DATA_FOLDER)
    df = pd.read_csv(os.path.join(DATA_FOLDER, TARGET_FILE))
    logger.info("Dataframe loaded!")
