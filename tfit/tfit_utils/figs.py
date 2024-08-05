# figs.py

import matplotlib.pyplot as plt
from collections.abc import Iterable

class fig:
    def __init__(self, *args, **kwargs):
        """
        Initialize the fig class. Currently, this class doesn't need initialization
        parameters, but this method can be extended in the future if needed.
        """
        pass

    @staticmethod
    def leg(*args, **kwargs):
        """Place a legend on the Axes. Wrapper for `plt.legend()`_

        Args:
            *args, **kwargs: args and kwargs passed to `plt.legend()`_
            ax (axes, optional): axes. If None, the current axes will be used. Default is None
            coord (tuple): (x, y) coordinates in data coordinates to place legend. 
                (x, y) places the corner of the legend specified by `loc` at x, y.
                If None, this is ignored. Default is None. Overwrites `bbox_to_anchor`.
            
        If not specified, the following parameters are passed to `ax.legend()`_:

        Args:
            labelspacing (float, optional): The vertical space between the legend 
                entries, in font-size units. Default is 0.1
            frameon (bool, optional): Whether the legend should be drawn on a patch 
                (frame). Default is False

        Returns:
            legend object

        .. _plt.legend(): https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        """
        # get axes
        ax = kwargs.pop('ax', plt.gca())

        # default attrs
        kwargs.setdefault('labelspacing', 0.1)
        kwargs.setdefault('frameon', False)

        # get coord
        coord = kwargs.pop('coord', None)

        # legend
        if coord is not None:   
            # check
            assert isinstance(coord, Iterable), 'coord must be a tuple of type (x, y)'
            assert len(coord) == 2, 'coord must be a tuple of type (x, y)'

            # convert to Data coordinates to axes coordinates
            _x, _y = ax.transData.transform(coord)
            x, y = ax.transAxes.inverted().transform((_x, _y)) 

            kwargs.setdefault('loc', "upper left")
            kwargs.setdefault('bbox_to_anchor', (x, y))

        return ax.legend(*args, **kwargs)