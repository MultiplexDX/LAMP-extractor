from functools import wraps
from time import time
from loguru import logger

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
            try:
                start = int(round(time() * 1000))
                return func(*args, **kwargs)
            finally:
                end = int(round(time() * 1000)) - start
                if end > 200:
                    logger.patch(lambda r: r.update(function=func.__name__, name=func.__module__)).warning(f"{end} ms")
    return _time_it

@measure
def gello():
    for u in range(200000):
        pass
    print("dskodksods")

#gello()
