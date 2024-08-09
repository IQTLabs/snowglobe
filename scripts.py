import uvicorn

def snowglobe_config():
    pass

def snowglobe_server(host='0.0.0.0', port=8000, log_level='info'):
    uvicorn.run('api:app', host=host, port=port, log_level=log_level)
