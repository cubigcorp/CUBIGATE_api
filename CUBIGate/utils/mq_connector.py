import pika
from configs.load_configs import load_configs
import json
from pyrabbit.api import Client


class MqConnector:
    def __init__(self):
        configs = load_configs()
        json_data = configs['rabbitmq']

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=json_data['host'], port=json_data['port'], virtual_host=json_data['vhost'], heartbeat=7200, 
                                      credentials=pika.PlainCredentials(json_data['username'], json_data['password'])))
        self.channel = self.connection.channel()

    def send_message(self, message, queue_name):
        if isinstance(message, dict):
            message = json.dumps(message)

        self.channel.queue_declare(queue=queue_name, durable=True)
        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,  # make message persistent
            ))
        print(" [x] Sent %r" % message)

    def close(self):
        self.connection.close()

    def get_list_connections(self, queue_name):
        res = self.channel.queue_declare(queue=queue_name, durable=True).method.consumer_count
        return res

