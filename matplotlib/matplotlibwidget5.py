from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from matplotlib import rcParams

rcParams['font.size'] = 9


class MatplotlibWidget5(Canvas):
    '''
    MatplotlibWidget inherits PyQt5.QWidgets
    and matplotlib.backend_bases.FigureCanvasBase
    
    Options: option_name (default_value)
    -------    
    parent (None): parent widget
    title ('')   : figure title
    xlabel ('')  : X-axis label
    ylabel ('')  : Y-axis label
    xlim (None)  : X-axis limits ([min, max])
    ylim (None)  : Y-axis limits ([min, max])
    xscale ('linear'): X-axis scale
    yscale ('linear'): Y-axis scale
    width (4) : width in inches
    height (3): height in inches
    dpi (100) : resolution in dpi
    hold (False): if False, figure will be cleared each time plot is called
    
    Widget attributes:
    -----------------
    figure: instance of matplotlib.figure.Figure
    axes: figure axes
    
    Example:
    -------
    self.widget = MatplotlibWidget(self, yscale='log', hold=True)
    from numpy import linspace
    x = linspace(-10, 10)
    self.widget.axes.plot(x, x**2)
    self.wdiget.axes.plot(x, x**3)
    '''

    def __init__(self, parent = None, title = '', xlabel = '', ylabel = '',
                 xlim = None, ylim = None, xscale = 'linear', yscale = 'linear',
                 width = 4, height = 3, dpi = 100):
        
        self.figure = Figure(figsize = (width, height), dpi = dpi)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if xscale is not None:
            self.axes.set_xscale(xscale)
        if yscale is not None:
            self.axes.set_yscale(yscale)
        if xlim is not None:
            self.axes.set_xlim(*xlim)
        if ylim is not None:
            self.axes.set_ylim(*ylim)

        super(MatplotlibWidget5, self).__init__(self.figure)
        self.setParent(parent)
        super(MatplotlibWidget5, self).setSizePolicy(QSizePolicy.Expanding, 
                                                    QSizePolicy.Expanding)
        super(MatplotlibWidget5, self).updateGeometry()

    def sizeHint(self):
        return QSize(*self.get_width_height())

    def minimumSizeHint(self):
        return QSize(10, 10)



# ==============================================================================
#   Example
# ==============================================================================
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QMainWindow, QApplication
    from numpy import linspace
    
    class ApplicationWindow(QMainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.mplwidget = MatplotlibWidget5(self, title = 'Example',
                                               xlabel = 'Linear scale',
                                               ylabel = 'Log scale',
                                               yscale = 'log')
            self.mplwidget.setFocus()
            self.setCentralWidget(self.mplwidget)
            self.plot(self.mplwidget.axes)
            
        def plot(self, axes):
            x = linspace(-10, 10)
            axes.plot(x, x**2)
            axes.plot(x, x**3)
        
    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
