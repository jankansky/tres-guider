import os
from PyQt5.QtGui import QIcon
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from matplotlib import rcParams
from matplotlibwidget5 import MatplotlibWidget5

rcParams['font.size'] = 9


class MatplotlibPlugin5(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        super(MatplotlibPlugin5, self).__init__(parent)
        self._initialized = False

    def initialize(self, editor):
        self._initialized = True

    def isInitialized(self):
        return self._initialized

    def createWidget(self, parent):
        return MatplotlibWidget5(parent)

    def name(self):
        return 'MatplotlibWidget'

    def group(self):
        return 'PyQt'

    def icon(self):
        return QIcon(os.path.join(rcParams['datapath'], 'images', 'matplotlib.png'))

    def toolTip(self):
        return ''

    def whatsThis(self):
        return ''

    def isContainer(self):
        return False

    def domXml(self):
        return '<widget class="MatplotlibWidget5" name="mplwidget">\n' \
               '</widget>\n'

    def includeFile(self):
        return 'matplotlibwidget5'



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = MatplotlibWidget5()
    widget.show()
    sys.exit(app.exec_())
