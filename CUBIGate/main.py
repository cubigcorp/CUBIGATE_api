from cubigate.generate import CubigDPGenerator
from cubigate.dp.utils.arg_utils import str2bool
import argparse
import os
import logging
from utils.mq_connector import MqConnector
from utils.db_connector import DBConnector
import pika
import sys
import json
import requests
import torch

logging.getLogger("pika").propagate = False

#fixed values for display
#generator = CubigDPGenerator()
db_connector = DBConnector()

#Just train the model to make DP-synthetic data


def train_data_generation_model(iterations=2, epsilon=1, delta=0, dataset="/var/dp_msv/datasets/cookie"):
    try:
        print(dataset)
        generator = CubigDPGenerator(gpu_num=0)
        data_checkpoint=generator.train(iterations, epsilon, delta, data_folder=dataset)
        print(data_checkpoint)
        del generator
        torch.cuda.empty_cache()
        return  data_checkpoint
    except Exception as e:
        raise e

#Just generate data with your data checkpoint (data checkpoint means model chekcpoint in Cubigate)
#Output: zip file of new data.
def generate_dp_data(base_data="./result/cookie/1/_samples.npz"):
    try:
        print(base_data)
        generator = CubigDPGenerator(gpu_num=1)
        print(3)
        new_data=generator.generate(base_data)
        del generator
        torch.cuda.empty_cache()
        print(type(new_data))
        return new_data
    except Exception as e:
        raise e
    
#Run the entire process of Cubigate (train, generate) => output is zip file of new image data
def train_generate_dp_data( iterations=2, epsilon=1, delta=0, dataset="/var/dp_msv/datasets/cookie" ):
    print(dataset)
    generator = CubigDPGenerator()
    data_checkpoint=generator.train(iterations, epsilon, delta, data_folder=dataset)
    new_data=generator.generate(data_checkpoint)
    print(type(new_data))
    return new_data

# Break down training into a series of detailed functions
# Each uses the output of the previous one 
# 1. Initialize -> variate
def initialize_training() -> str:
    initial = generator.initialize()
    return initial

# 2. Variate -> measure
def variate_prev_data(previous: str) -> str:
    generator.initialize()  
    variated = generator.variate(samples_path=previous)
    return variated

# 3. measure -> select
def measure_variated(variated: str, epsilon: float, delta: float) -> str:
    generator.initialize()  
    measured = generator.measure(samples_path=variated, epsilon=epsilon, delta=delta)
    return measured


# 4. select -> variate
def select_measured(measured: str, variated: str) -> str:
    selected = generator.select(dist_path=measured, samples_path=variated)
    return selected


def on_message_callback_train(ch, method, properties, body):
    try:
        print(f" [x] Received {body.decode()}")
        data = body.decode()
        data=json.loads(data)
        job_id = data['job_id']
        user_id = data['user_id']
        service_id = data['service_id']
        
        dataset=data["dataset"]

        job_status = db_connector.call_stored_procedure('service_request_check_status', params=[job_id], fetch_all=False)
        if job_status['status'] == 'SUCCESS' or job_status['status'] == 'FAILED':
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # Get callback URL
        callback_url = db_connector.execute_query('service_request_get_callback_url', params=[job_id], fetch_all=False)
        logging.info(f"Job id: {job_id} - Callback URL: {callback_url}")
        
        empty_json_str = json.dumps({})
        
        # Update job status to executing
        db_connector.execute_query('service_request_update', params=[job_id, 'EXECUTING', empty_json_str, ''], fetch_all=False)
        
        
        result_file_path = train_data_generation_model(data['iterations'], data['epsilon'], data['delta'], dataset)
        
        result = {"job_id": job_id, "result": {"file_path": result_file_path}}
        # send result to callback url
        if callback_url:
            requests.post(callback_url, json=result)
        
        logging.info(f"Result file path: {result_file_path}")
        
        result_str = json.dumps(result)
        # Update job status to success
        db_connector.execute_query('service_request_update', params=[job_id, 'SUCCESS', result_str, ''], fetch_all=False)
        db_connector.execute_query('user_service_update_status', params=[user_id, service_id, 'RUNNING'])
        
        # Update generate service to waiting
        db_connector.execute_query('user_service_update_status', params=[user_id, 2, 'WAITING'])
        
        #ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        # Update job status to error
        db_connector.execute_query('service_request_update', params=[job_id, 'FAILED', empty_json_str, str(e)], fetch_all=False)
        db_connector.execute_query('user_service_update_status', params=[user_id, service_id, 'ERROR'])
    finally:
        logging.info(f" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)


##FOR TEST before deploy
def test_train():
    
    print(f" [x] DEV train")

    dataset="/var/dp_msv/datasets/lfw"
    
    print(dataset)
    result_file_path = train_data_generation_model(2,1, 0.01, dataset)
    print(result_file_path)
    print("done")
    
def test_generate():

    checkpoint_path=""
    
    file_name = generate_dp_data(checkpoint_path).filename

    print(file_name)
    print("done")

      
def on_message_callback_generate(ch, method, properties, body):
    try:
        print(f" [x] Received {body.decode()}")
        data = body.decode()
        data=json.loads(data)
        job_id = data['job_id']
        user_id = data['user_id']
        service_id = data['service_id']
        
        job_status = db_connector.call_stored_procedure('service_request_check_status', params=[job_id], fetch_all=False)
        if job_status['status'] == 'SUCCESS' or job_status['status'] == 'FAILED':
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        empty_json_str = json.dumps({})
        
        # Get callback URL
        callback_url = db_connector.execute_query('service_request_get_callback_url', params=[job_id], fetch_all=False)
        logging.info(f"Job id: {job_id} - Callback URL: {callback_url}")
        
        # Update job status to executing
        db_connector.execute_query('service_request_update', params=[job_id, 'EXECUTING', empty_json_str, ''], fetch_all=False)
        
        print(data["checkpoint_path"])
        
        file_name = generate_dp_data(data['checkpoint_path']).filename
        
        result = {"job_id": job_id, "result": {"file_path": file_name}}
        # send result to callback url
        if callback_url:
            requests.post(callback_url, json=result)
        
        logging.info(f"Result file path: {file_name}")
        
        result_str = json.dumps(result)
        # Update job status to success
        db_connector.execute_query('service_request_update', params=[job_id, 'SUCCESS', result_str, ''], fetch_all=False)
        db_connector.execute_query('user_service_update_status', params=[user_id, service_id, 'FINISHED'])
        #ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        # Update job status to error
        db_connector.execute_query('service_request_update', params=[job_id, 'FAILED', empty_json_str, str(e)], fetch_all=False)
        db_connector.execute_query('user_service_update_status', params=[user_id, service_id, 'ERROR'])
    finally:
        logging.info(f" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        #torch.cuda.empty_cache()
        
def on_message_callback_clear_messages(ch, method, properties, body):
    data = body.decode()
    data=json.loads(data)
    job_id = data['job_id']
    empty_json_str = json.dumps({})
    db_connector.execute_query('service_request_update', params=[job_id, 'KILLED', empty_json_str, ''], fetch_all=False)
    ch.basic_ack(delivery_tag=method.delivery_tag)



mode = sys.argv[1]
if not mode or mode not in ['train', 'generate', 'DEV_train', "DEV_generate"]:
    print("Please specify mode: train or generate")
    exit(1)

while True:
    if mode == 'train':
        queue_name = 'dp_msv_train'
    elif mode=="generate":
        print("generate")
        queue_name = 'dp_msv_generate'
    else:
        pass
    try:
        if mode=="DEV_train":
            test_train()
        elif mode=="DEV_generate":
            test_generate()
        else:
            logging.info(f" [*] Waiting for messages in queue: {queue_name}. To exit press CTRL+C")
            mq_connector = MqConnector()
            channel = mq_connector.channel
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_qos(prefetch_count=1)
            if queue_name == 'dp_msv_train':
                channel.basic_consume(queue_name, on_message_callback_train)
            else:
                channel.basic_consume(queue_name, on_message_callback_generate)
            channel.start_consuming()
        #connection = mq_connector.connection
        #connection.process_data_events()
    # Don't recover if connection was closed by broker
    except pika.exceptions.ConnectionClosedByBroker:
        break
    # Don't recover on channel errors
    except pika.exceptions.AMQPChannelError:
        break
    # Recover on all other connection errors
    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f'Error: {str(e)}')
        continue
