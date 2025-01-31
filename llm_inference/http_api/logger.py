from loguru import logger
import sys
import traceback

corr_id = 'BASIC'

def add_traceback(record):
    global corr_id
    extra = record["extra"]
    # extra['corr_id'] = f"CID: {corr_id}\t" if corr_id else ''
    if extra.get("with_traceback", False):
        extra["traceback"] = "\t" + "\t".join([i.replace('\n', '') for i in traceback.format_stack()])
    else:
        extra["traceback"] = ""

logger.remove()
logger = logger.patch(add_traceback).bind(corr_id=corr_id)
logger.add(sys.stdout, format="[{level.icon}  {level.name[0]}]\tCID: {extra[corr_id]}\t{message}{extra[traceback]}")    # {time}

if __name__ == "__main__":
    logger.warning("This is a module and should not be run directly. Running self-tests...")
    logger.info("Okay!")
