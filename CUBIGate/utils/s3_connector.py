import boto3
from configs.load_configs import load_configs
import tempfile
import logging
import datetime
import pytz

LOGGER = logging.getLogger(__name__)

class S3Connector:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            configs = load_configs()
            configs = configs['s3']

            service_name = 's3'
            endpoint_url = configs["endpoint_url"]
            region_name = configs["region_name"]
            access_key = configs["access_key"]
            secret_key = configs["secret_key"]
            bucket_name = configs["bucket_name"]

            cls.__instance = super(S3Connector, cls).__new__(cls)
            cls.__instance.s3_client = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key, region_name=region_name)
            cls.__instance.s3_resource = boto3.resource(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key, region_name=region_name)
            cls.__instance.bucket_name = bucket_name
        return cls.__instance

    def upload_file(self, s3_key, binary_data, target_folder='all/'):
        try:
            folder_name = target_folder
            self.s3_client.put_object(Bucket=self.bucket_name, Key=f"{folder_name}{s3_key.split('/')[1]}/")
            self.s3_resource.Object(self.bucket_name, s3_key).put(Body=binary_data)
            LOGGER.info(f"File uploaded to S3: {s3_key}")
        except Exception as e:
            LOGGER.error(f"Error uploading file to S3: {e}")
            raise Exception(f"Error uploading file to S3: {e}")

    def download_file(self, s3_key, local_file_path):
        try:
            self.s3_client.download_fileobj(self.bucket_name, s3_key, local_file_path)
        except Exception as e:
            LOGGER.error(f"Error downloading file from S3: {e}")
            raise Exception(f"Error downloading file from S3: {e}")

    def delete_file(self, s3_key):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            LOGGER.info(f"File deleted from S3: {s3_key}")
        except Exception as e:
            LOGGER.error(f"Error deleting file from S3: {e}")
            raise Exception(f"Error deleting file from S3: {e}")

    def copy_file(self, s3_key, target_s3_key):
        try:
            self.s3_client.copy_object(Bucket=self.bucket_name, CopySource=f"{self.bucket_name}/{s3_key}", Key=target_s3_key)
            LOGGER.info(f"File copied from S3: {s3_key} to {target_s3_key}")
        except Exception as e:
            LOGGER.error(f"Error copying file from S3: {e}")
            raise Exception(f"Error copying file from S3: {e}")

    def check_file_exists(self, s3_key):
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception as e:
            return False
