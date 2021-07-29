import sys
import signal
from pathlib import Path
import os
import logging
from functools import partial
from tqdm import tqdm as std_tqdm
import numpy as np
import tensorflow as tf
import tfplot
import gin

tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

def setup_logger(log_dir,args):
    # unmask SIGIGN; (condor mask SIGINT for child processes)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    Path(log_dir).mkdir(parents=True,exist_ok=True)

    assert 'temp' in log_dir or not os.path.exists(os.path.join(log_dir,'args.txt')), f'{log_dir} exist!'

    with open(os.path.join(log_dir,'args.txt'),'w') as f:
        f.write(str(args))

    # TF Summary Writer as Logger
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,'tb'))
    class TFSummaryHandler(logging.StreamHandler):
        def emit(self, record):
            with summary_writer.as_default():
                if record.msg == 'raw':
                    tag, value, it = record.args
                    tf.summary.scalar(tag, value, step=it)
                elif record.msg == 'histogram':
                    tag, value, it = record.args
                    tf.summary.histogram(tag, value, step=it)
                elif record.msg == 'text':
                    tag, value, it = record.args
                    tf.summary.text(tag, value, step=it)
                elif record.msg == 'img':
                    tag, value, it = record.args
                    value = tfplot.figure.to_array(value)
                    if value.ndim == 3:
                        value = value[None]
                    tf.summary.image(tag, value, step=it, max_outputs=3)
                else:
                    summary_str, it = record.args

        def flush(self):
            summary_writer.flush()

    handler = TFSummaryHandler()
    handler.setLevel(logging.DEBUG)

    logger = logging.getLogger('summary_writer')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Standard stdout Logger
    np.set_printoptions(precision=4,suppress=True)

    class TqdmHandler(logging.Handler):
        def emit(self, record):
            try:
                std_tqdm.write(self.format(record),file=sys.stderr)
                #with summary_writer.as_default():
                #    tf.summary.text('stdout', self.format(record), step=0)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    handler = TqdmHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(f'[{log_dir}] %(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger('stdout')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

def write_gin_config(log_dir):
    # write down gin config
    # CAUTION: call this function after all operative configs are in effect.
    summary_writer = logging.getLogger('summary_writer')

    with open(os.path.join(log_dir,'config.gin'),'w') as f:
        f.write(gin.operative_config_str())
    summary_writer.info('text','gin/config',gin.operative_config_str(),0)

def parse_event(path, capture_tags):
    # path: events.out file
    from collections import defaultdict
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    metrics = defaultdict(list)
    for e in summary_iterator(path):
        wall_time = e.wall_time
        step = e.step

        for v in e.summary.value:
            if not v.tag in capture_tags: continue

            # To see all possible data type in Value, see
            # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/summary.proto
            metrics['step'].append(step)
            metrics[v.tag].append(tf.make_ndarray(v.tensor))

    metrics_df = pd.DataFrame({k: v for k,v in metrics.items() if len(v) > 1})
    return metrics_df
