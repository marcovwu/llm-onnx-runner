def custom_tqdm(input, _):
    return input


class Logger:
    def info(text):
        print(text)

    def warning(text):
        print(text)

    def error(text):
        print(text)

    def debug(text):
        print(text)


# import tqdm
try:
    from tqdm import tqdm
    tqdm = tqdm
except Exception:
    tqdm = custom_tqdm


# import logger
try:
    from loguru import logger
    logger = logger
except Exception:
    logger = Logger