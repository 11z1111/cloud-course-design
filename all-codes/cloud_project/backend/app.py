import os
import redis
from flask import Flask, jsonify

app = Flask(__name__)

# 从环境变量读取Redis配置
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
redis_password = os.getenv('REDIS_PASSWORD', None)

# 连接Redis
try:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True
    )
    redis_client.ping()
    redis_connected = True
except:
    redis_connected = False
    print("Warning: Redis connection failed")

@app.route('/api/ping', methods=['GET'])
def ping():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'redis_connected': redis_connected
    })

@app.route('/api/hello', methods=['GET'])
def hello():
    """测试接口"""
    return jsonify({
        'message': 'Hello from Flask backend!'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)