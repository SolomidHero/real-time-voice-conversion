from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QLabel
from PySide2.QtCore import Signal

import io

def get_figure_widget(fig):
  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)

  pixmap = QPixmap()
  pixmap.loadFromData(buf.read())
  buf.close()

  fig_label = QLabel()
  fig_label.setPixmap(pixmap)

  return fig_label


def get_figure_pixmap(fig):
  buf = io.BytesIO()
  fig.savefig(buf, format='png', transparent=True)
  buf.seek(0)

  pixmap = QPixmap()
  pixmap.loadFromData(buf.read())
  buf.close()

  return pixmap



class FigureWidget(QLabel):
  draw_signal = Signal()

  def __init__(self, fig, axis):
    super().__init__()
    self.fig = fig
    self.axis = axis
    self.draw_signal.connect(self.draw)

  def update(self):
    self.draw_signal.emit()

  def draw(self):
    self.setPixmap(get_figure_pixmap(self.fig))

