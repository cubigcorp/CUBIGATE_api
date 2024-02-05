import mariadb
from configs.load_configs import load_configs


class DBConnector:
    _instance = None

    def __new__(cls):
        configs = load_configs()

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.host = configs['db']['host']
            cls._instance.username = configs['db']['username']
            cls._instance.password = configs['db']['password']
            cls._instance.database = configs['db']['name']
            cls._instance.port = configs['db']['port']
            cls._instance.connection = None
            cls._instance.cursor = None
        return cls._instance

    def execute_query(self, function_name, params=[], fetch_all=True):
        connection = None
        cursor = None
        try:
            connection = mariadb.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            cursor = connection.cursor()
            query = f"SELECT {function_name}({','.join(['%s'] * len(params))})"
            cursor.execute(query, params)
            connection.commit()
            if fetch_all:
                return cursor.fetchall()
            else:
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    return {}
        except mariadb.Error as e:
            raise Exception(f"Error executing query: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def call_stored_procedure(self, procedure_name, params=None, fetch_all=True):
        connection = None
        cursor = None
        try:
            connection = mariadb.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            cursor = connection.cursor()
            if params:
                cursor.callproc(procedure_name, params)
            else:
                cursor.callproc(procedure_name)

            columns = cursor.description
            if not columns:
                return {}

            if fetch_all:
                result = [{columns[index][0]: column for index, column in enumerate(value)} for value in
                          cursor.fetchall()]
            else:
                result = [{columns[index][0]: column for index, column in enumerate(value)} for value in
                          cursor.fetchall()]
                if result:
                    result = result[0]
                else:
                    result = {}
                
            return result
        except mariadb.Error as e:
            raise Exception(f"Error calling stored procedure: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
