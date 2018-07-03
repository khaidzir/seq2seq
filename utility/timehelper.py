'''
-------------------- TIME RELATED HELPER FUNCTION --------------------

helper function for time related operation (mainly used for logging)

----------------------------------------------------------------------
'''

import time
import math

# converting second to minute
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# getting current time and estimating current training progress
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return ('%s (- %s)' % (asMinutes(s), asMinutes(rs)))