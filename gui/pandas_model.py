from PyQt6.QtCore import QAbstractTableModel, Qt
import pandas as pd

class PandasTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data if data is not None else pd.DataFrame()

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                try:
                    return str(self._data.iloc[index.row(), index.column()])
                except IndexError:
                    return None # Should not happen with valid index from rowCount/columnCount
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                try:
                    return str(self._data.columns[section])
                except IndexError:
                    return None
            if orientation == Qt.Orientation.Vertical:
                try:
                    return str(self._data.index[section])
                except IndexError:
                    return None
        return None
    
    def setData(self, data_frame: pd.DataFrame): # Custom method to update data
        self.beginResetModel()
        self._data = data_frame.copy() # Use a copy
        self.endResetModel()
