import logging
import sys


class StdLogger():
    def __init__(self, file_path = "", level=logging.INFO):
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                      datefmt="%H:%M:%S")
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(level)

        if file_path:
            file_hander = logging.FileHandler(file_path)
            file_hander.setFormatter(formatter)
            self.logger.addHandler(file_hander)

        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)





# cfg = OmegaConf.load(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.yaml'))
# log_dir = os.path.join(cfg.logger.log_dir, str(int(time())))

# file_path = os.path.join(log_dir, "train.log")

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

Logger = StdLogger("",
                   level=logging.INFO).logger