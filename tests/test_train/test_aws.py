import pytest

from train.aws import fetch_s3_data, dump_s3_data
from train.exceptions import InvalidBucketName, MissingCredentials


class TestAWS(object):
    def test_fetch_s3_data_RaisesExceptionInvalidBucketName(self) -> None:
        with pytest.raises(InvalidBucketName):
            fetch_s3_data(
                bucket_name="",
                target_folder="data",
                aws_access_key="1234",
                aws_secret_key="1234",
            )

    def test_fetch_s3_data_RaisesExceptionInvalidCredentials(self) -> None:
        with pytest.raises(MissingCredentials):
            fetch_s3_data(
                bucket_name="data-cicd-gfluz94",
                target_folder="data",
                aws_access_key=None,
                aws_secret_key=None,
            )

    def test_dump_s3_data_RaisesExceptionInvalidBucketName(self) -> None:
        with pytest.raises(InvalidBucketName):
            dump_s3_data(
                local_filepath="output/model.pkl",
                bucket_name="",
                key_name="model",
                aws_access_key="1234",
                aws_secret_key="1234",
            )

    def test_dump_s3_data_RaisesExceptionInvalidCredentials(self) -> None:
        with pytest.raises(MissingCredentials):
            dump_s3_data(
                local_filepath="output/model.pkl",
                bucket_name="data-cicd-gfluz94",
                key_name="model",
                aws_access_key=None,
                aws_secret_key=None,
            )