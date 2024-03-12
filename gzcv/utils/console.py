import logging
import sys
import traceback
from functools import wraps

import rich


def pretty(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        console = rich.console.Console()
        try:
            ret = func(*args, **kwargs)
            return ret
        except KeyboardInterrupt:
            logger.exception("Keyboard Interrupt")
            sys.exit(1)
        except Exception:
            logger.exception(traceback.format_exc())
            console.print_exception()
            sys.exit(1)

    return wrapper
