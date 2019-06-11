import datetime as dt
import time as t

import mltools.utilities as util

LOGTAG = "LOG"


class Formatting:
    BOLD = "\033[1m"
    UNDERLINE = '\033[4m'
    REVERSE = "\033[;7m"
    WHITE = "\033[30m"
    RED = "\033[31m"
    YELLOW = "\033[32m"
    FAINT_YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PINK = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[37m"
    B_WHITE = "\033[30m"
    B_RED = "\033[31m"
    B_YELLOW = "\033[32m"
    B_FAINT_YELLOW = "\033[33m"
    B_BLUE = "\033[34m"
    B_PINK = "\033[35m"
    B_CYAN = "\033[36m"
    B_GREY = "\033[37m"
    RESET = "\033[0m"


class Level(util.OrderedEnum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    HIGHLIGHT = 5


LEVEL_FORMATTING = {Level.DEBUG: '', Level.INFO: '', Level.WARNING: Formatting.FAINT_YELLOW,
                    Level.ERROR: Formatting.RED, Level.HIGHLIGHT: Formatting.REVERSE}
LEVEL_HEADER = {Level.DEBUG: 'DEBUG.',
                Level.INFO: '{0}INFO.{1}'.format(Formatting.CYAN, Formatting.RESET),
                Level.WARNING: '{0}WARN!{1}'.format(Formatting.YELLOW + Formatting.UNDERLINE, Formatting.RESET),
                Level.ERROR: '{0}ERROR!{1}'.format(Formatting.RED + Formatting.BOLD, Formatting.RESET),
                Level.HIGHLIGHT: '{0}HIGH.{1}'.format(Formatting.REVERSE + Formatting.BOLD, Formatting.RESET)}

TAGS = {'<b>': Formatting.BOLD, '<u>': Formatting.UNDERLINE, '<r>': Formatting.REVERSE}

verbosity = Level.INFO
save_path = None


def log(message, log_tag='', level=Level.INFO, temporary = False):
    for tag, formatting_code in TAGS.items():
        message_raw = message.replace(tag, '')
        message = message.replace(tag, LEVEL_FORMATTING[level] + formatting_code)
        end_tag = tag[:1] + "/" + tag[1:]
        message = message.replace(end_tag, Formatting.RESET + LEVEL_FORMATTING[level])
        message_raw = message_raw.replace(end_tag, '')

    if save_path is not None:
        with open(save_path, 'a+') as log_file:
            log_file.write(dt.datetime.fromtimestamp(t.time()).strftime('%Y-%m-%d %H:%M:%S')
                           + ' -- ' + message_raw + '\n')

    if level >= verbosity:
        print("{1}[{2}] {0}{3}{1} {4}{0}".format(Formatting.RESET, LEVEL_FORMATTING[level], log_tag,
                                                 LEVEL_HEADER[level], message), end='\r' if temporary else '\n')


def setup_logging(verbosity_=Level.INFO, path_=None):
    global verbosity, save_path
    verbosity, save_path = verbosity_, path_
    if path_ is not None:
        log("Will be writing a log of all messages in file {0}.".format(save_path), LOGTAG, Level.INFO)
