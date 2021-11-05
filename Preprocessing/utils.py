import preprocessor as p
import numpy as np
import emoji
import re

from datetime import datetime

BAD_SYMBOLS_RE = re.compile('[/(){}\[\]\|@_\+\-:*]')


def str_to_date(s):
    date_time = datetime.strptime(s, '%a %b %d %H:%M:%S %z %Y')
    return date_time


def text_processing(s):
    p.set_options(
        p.OPT.URL,  # Remove URL
        p.OPT.MENTION,  # Remove @
        p.OPT.HASHTAG,  # Remove #
        p.OPT.RESERVED,  # Remove RT and FAV
        p.OPT.SMILEY  # Remove Smileies
    )
    s = p.clean(s)

    s = emoji.demojize(s, delimiters=(' ', ' '))
    s = re.sub(BAD_SYMBOLS_RE, ' ', s)
    s = s.strip()

    return s


def empty_str_to_na(s):
    if s == '':
        return np.nan
    return s
