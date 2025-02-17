import os
import subprocess
from contextlib import contextmanager

@contextmanager
def set_proxy():
    result = subprocess.run(
        'bash -c "source /etc/network_turbo && env | grep proxy"', 
        shell=True, capture_output=True, text=True
    )

    proxies = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
            
    yield

    for key, value in os.environ.items():
        if 'proxy' in key:
            os.environ.pop(key, None)  # Remove the key if it wasn't set originally