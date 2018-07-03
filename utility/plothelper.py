'''
-------------------- PLOT RELATED HELPER FUNCTION --------------------

helper function for plot related operation
mainly used for plotting change in loss value for each iterations

----------------------------------------------------------------------
'''

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# showing simple 2d plot
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    #plt.show()