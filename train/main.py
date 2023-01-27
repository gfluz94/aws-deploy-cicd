import os
from argparse import ArgumentParser
from dotenv import load_dotenv
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from aws import fetch_s3_data
from model import TrainingOrchestrator
from exceptions import EnvironmentVariablesMissing

TARGET_FILE = "dataset.csv"


if __name__ == "__main__":
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
    parser.add_argument(
        "--target-col-name",
        metavar="N",
        type=str,
        help="Name of the target variable in input data.",
        default="default",
    )
    parser.add_argument(
        "--user-id-col-name",
        metavar="N",
        type=str,
        help="Name of the unique applicant identifiet in input data.",
        default="uuid",
    )
    parser.add_argument(
        "--selected-features",
        metavar="N",
        type=str,
        nargs="+",
        help="Selected features of interest from raw dataset.",
        default=[
            "max_paid_inv_0_24m",
            "avg_payment_span_0_12m",
            "sum_capital_paid_account_0_12m",
            "time_hours",
            "recovery_debt",
            "sum_capital_paid_account_12_24m",
            "num_active_div_by_paid_inv_0_12m",
            "sum_paid_inv_0_12m",
            "account_days_in_rem_12_24m",
            "num_arch_ok_0_12m",
            "account_amount_added_12_24m",
            "merchant_group",
            "has_paid",
            "account_status",
            "account_worst_status_0_3m",
            "account_worst_status_12_24m",
            "account_worst_status_3_6m",
            "account_worst_status_6_12m",
            "status_last_archived_0_24m",
            "status_2nd_last_archived_0_24m",
            "status_3rd_last_archived_0_24m",
            "status_max_archived_0_6_months",
            "status_max_archived_0_12_months",
            "status_max_archived_0_24_months",
        ],
    )
    parser.add_argument(
        "--cols-to-apply-log",
        metavar="N",
        type=str,
        nargs="+",
        help="Features that require log transformation.",
        default=["max_paid_inv_0_24m", "sum_capital_paid_account_0_12m"],
    )
    parser.add_argument(
        "--merchant-groups",
        metavar="N",
        type=str,
        nargs="+",
        help="Set of merchant groups of interest.",
        default=[
            "Clothing & Shoes",
            "Intangible products",
            "Food & Beverage",
            "Erotic Material",
            "Entertainment",
        ],
    )
    parser.add_argument(
        "--test-size",
        metavar="N",
        type=float,
        help="Fraction of dataset that will be used for test purposes.",
        default=0.25,
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Whether or not to calibrate XGBoost Classifier.",
        default=False,
    )
    parser.add_argument(
        "--n-folds",
        metavar="N",
        type=int,
        help="Number of folds for calibration.",
        default=3,
    )
    # XGBOOST PARAMS
    parser.add_argument(
        "--xgboost-n-estimators",
        metavar="N",
        type=int,
        help="Number of estimators for XGBoost Classifier.",
        default=200,
    )
    parser.add_argument(
        "--xgboost-min-child-weight",
        metavar="N",
        type=int,
        help="Minimum child weight for XGBoost Classifier.",
        default=5,
    )
    parser.add_argument(
        "--xgboost-max-depth",
        metavar="N",
        type=int,
        help="Maximum depth for XGBoost Classifier.",
        default=4,
    )
    parser.add_argument(
        "--xgboost-subsample",
        metavar="N",
        type=float,
        help="Subsample for XGBoost Classifier.",
        default=1.0,
    )
    parser.add_argument(
        "--xgboost-lr",
        metavar="N",
        type=float,
        help="Learning rate for XGBoost Classifier.",
        default=0.026826957952797246,
    )
    parser.add_argument(
        "--xgboost-gamma",
        metavar="N",
        type=float,
        help="Gama for XGBoost Classifier.",
        default=1.5,
    )
    parser.add_argument(
        "--xgboost-colsample-bytree",
        metavar="N",
        type=float,
        help="Column sample by tree for XGBoost Classifier.",
        default=1.0,
    )
    parser.add_argument(
        "--save-evaluation-artifacts",
        action="store_true",
        help="Whether or not to save evaluation output.",
        default=False,
    )
    parser.add_argument(
        "--evaluation-artifacts-path",
        metavar="N",
        type=str,
        help="Path where evluation artifacts should be dumped to.",
        default="output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether or not to display information on terminal.",
        default=False,
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        help="Random seed to ensure reproducibility.",
        default=99,
    )
    args = parser.parse_args()
    load_dotenv()
    if args.verbose:
        logger.info("Parameters defined!")

    if args.verbose:
        logger.info("Finding and reading raw data...")
    DATA_FOLDER = os.path.join(os.curdir, args.data_folder)
    FILES_IN_FOLDER = os.listdir(DATA_FOLDER)
    if TARGET_FILE not in FILES_IN_FOLDER:
        if args.verbose:
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
        if args.verbose:
            logger.info(
                "Dataset successfully uploaded to local folder %s...", DATA_FOLDER
            )
    df = pd.read_csv(os.path.join(DATA_FOLDER, TARGET_FILE), sep=";")
    if args.verbose:
        logger.info("Dataframe loaded!")

    if args.verbose:
        logger.info("Starting training loop...")
    xgboost_params = {
        "n_estimators": args.xgboost_n_estimators,
        "min_child_weight": args.xgboost_min_child_weight,
        "max_depth": args.xgboost_max_depth,
        "subsample": args.xgboost_subsample,
        "learning_rate": args.xgboost_lr,
        "gamma": args.xgboost_gamma,
        "colsample_bytree": args.xgboost_colsample_bytree,
    }
    training_orchestrator = TrainingOrchestrator(
        target_col_name=args.target_col_name,
        unique_id_col_name=args.user_id_col_name,
        selected_cols=args.selected_features,
        log_transform_cols=args.cols_to_apply_log,
        merchant_groups=args.merchant_groups,
        test_set_size=args.test_size,
        calibrate=args.calibrate,
        n_folds=args.n_folds,
        xgboost_parameters=xgboost_params,
        save_eval_artifacts=args.save_evaluation_artifacts,
        eval_artifacts_path=args.evaluation_artifacts_path,
        verbose=args.verbose,
        random_seed=args.seed,
    )
    model = training_orchestrator.fit(df)
    if args.verbose:
        logger.info("Evaluating model's performance...")
    (train_metrics, train_bands), (
        test_metrics,
        test_bands,
    ) = training_orchestrator.evaluate_performance(df, threshold=0.50, show_viz=args.verbose)
    if args.verbose:
        logger.info("Performance evaluation finished!")

    #EXPORT PRINT BANDS/METRICS

    #EXPORT MODEL TO S3
