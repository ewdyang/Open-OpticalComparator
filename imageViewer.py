import sys
from typing import Optional
from PySide6.QtWidgets import *
from PySide6.QtCore import QObject, Qt, Signal, Slot, QStandardPaths, QDir, QThread, QEvent, QMargins, QSize, QPointF, QTimer
from PySide6.QtGui import (
    QPalette,
    QColor,
    QGuiApplication,
    QImageReader,
    QImage,
    QImageWriter,
    QAction,
    QKeySequence,
    QPixmap,
    QMouseEvent,
    QResizeEvent,
    QValidator
    )
import cv2
import numpy as np
import trimesh
import json
import datetime
import inspect
import os
from ImageOperations import *
from QTImageManipulator import QTImageManipulator

class ImageScrollArea(QScrollArea):
    mouseCoordSignal = Signal(str)
    mouseClickSignal = Signal(object)

    def __init__(self):
        super().__init__()
        self.scaleFactors = (1.0, 1.0)

        self.imageLabel = QLabel()
        # self.imageLabel.setBackgroundRole(QPalette.ColorRole.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Ignore size hint, use as much space as possible
        self.imageLabel.setScaledContents(True) # Makes Pixmap fill label
        self.imageLabel.setMouseTracking(True)
        self.imageLabel.resizeEvent = self.labelResizeEvent
        self.resize(800,600)

        self.setBackgroundRole(QPalette.ColorRole.Dark)
        self.setWidget(self.imageLabel)
        self.setVisible(False)

        def mouseTracking(event: QMouseEvent):
            pos = (event.position().x() * self.scaleFactors[0], event.position().y() * self.scaleFactors[1])
            self.mouseCoordSignal.emit(f'X:{pos[0]:.2f}, Y:{pos[1]:.2f}')
            event.ignore()

        def mouseClick(event: QMouseEvent):
            pos = (round(event.position().x() * self.scaleFactors[0]), round(event.position().y() * self.scaleFactors[1]))
            # print(pos)
            self.mouseClickSignal.emit(pos)
            event.ignore()

        self.imageLabel.mouseMoveEvent = mouseTracking
        self.imageLabel.mousePressEvent = mouseClick

    def calculate_scale(self):
        current_size = self.imageLabel.size()
        original_size = self.imageLabel.sizeHint()
        scaleX = original_size.width() / current_size.width()
        scaleY = original_size.height() / current_size.height()
        self.scaleFactors = (scaleX, scaleY)

    def labelResizeEvent(self, arg__1: QResizeEvent) -> None:
        self.calculate_scale()
        return super(QLabel,self.imageLabel).resizeEvent(arg__1) # Hack without having to subclass QLabel and set up signal/slots

    def adjust_scrollbar(self, scrollBar, factor):
        scrollBar.setValue(int(scrollBar.value() / factor + (1/factor - 1) * scrollBar.pageStep() / 2))    

    def scale_image(self, factor):
        self.scaleFactors = tuple(element * factor for element in self.scaleFactors)
        new_size = QSize(self.imageLabel.pixmap().size().width() / self.scaleFactors[0], self.imageLabel.pixmap().size().height() / self.scaleFactors[1])
        self.imageLabel.resize(new_size)

        self.adjust_scrollbar(self.horizontalScrollBar(), factor)
        self.adjust_scrollbar(self.verticalScrollBar(), factor)

    @Slot()
    def set_image_slot(self, image: QImage):
        old_size = self.imageLabel.sizeHint()
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        if self.imageLabel.sizeHint() != old_size:
            self.scale_image(1)
        self.setVisible(True)
    
    def reset_image_size(self):
        self.imageLabel.adjustSize()

    @Slot()
    def zoom_in_slot(self):
        if self.imageLabel.size().width() / 0.8 > 50000 or self.imageLabel.size().height() / 0.8 > 50000:
            return
        self.scale_image(0.8)
        # TODO Fix crash when zooming in too far(due to imageLabel being too large)
    
    @Slot()
    def zoom_out_slot(self):
        if self.imageLabel.size().width() / 1.25 < 10 or self.imageLabel.size().height() / 1.25 < 10:
            return
        self.scale_image(1.25)
    
    @Slot()
    def normal_size_slot(self):
        self.imageLabel.adjustSize()
        # self.scaleFactors = (1.0, 1.0)

    @Slot()
    def stretch_to_window_slot(self, isChecked: bool):
        self.setWidgetResizable(isChecked)
        if(not isChecked):
            self.normal_size_slot()

    @Slot()
    def ratio_fit_to_window(self):
        # self.scaleFactors = self.imageLabel.size().width() / self.imageLabel.sizeHint().width() #Scaling still not working
        self.imageLabel.resize(self.imageLabel.pixmap().size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio).shrunkBy(QMargins(1,1,1,1)))
        pass
        # TODO implement function to rescale to fit window while keeping aspect ratio
        

class MainWindow(QMainWindow):
    load_new_image = Signal(str)
    save_image_file = Signal(str)
    update_view_list = Signal()
    update_param_list = Signal()
    add_operation_page = Signal(object)
    update_operation_limits = Signal(object)
    update_all_operation_limits = Signal()
    add_operation = Signal(object)
    reset_operations = Signal()
    set_latest_input_image = Signal(object)
    update_active_image = Signal(object)
    apply_operation = Signal(object)
    apply_seq_operations = Signal()
    save_workflow_json = Signal(object)
    add_workflow_from_json = Signal(object)
    load_workflow_json = Signal(object)
    replicate_active_operation = Signal()

    def __init__(self):
        super().__init__()

        self.zoomInAct = QAction()
        self.zoomOutAct = QAction()
        self.normalSizeAct = QAction()
        self.stretchToWindowAct = QAction()

        self.image = QImage()

        self.setWindowTitle("Image Viewer")
        
        self.imageArea = ImageScrollArea()
        self.centreLayout = QHBoxLayout()
        self.centreLayout.addWidget(self.imageArea)
        self.centreWidget = QWidget(self)
        self.centreWidget.setLayout(self.centreLayout)

        self.image_thread = QThread()
        self.image_manipulator = QTImageManipulator()
        self.image_manipulator.moveToThread(self.image_thread)
        
        self.select_view_widget = SelectViewWidget(self.image_manipulator)
        self.parameter_widget = ParameterAdjustWidget(self.image_manipulator)

        #Setting up signals for each of the slots on image_manipulator
        self.load_new_image.connect(self.image_manipulator.slot_load_image)
        self.save_image_file.connect(self.image_manipulator.slot_save_active_results)
        self.add_operation.connect(self.image_manipulator.slot_add_operation)
        self.set_latest_input_image.connect(self.image_manipulator.set_input_image_to_latest)
        self.update_operation_limits.connect(self.image_manipulator.slot_update_operation_limits)
        self.update_all_operation_limits.connect(self.image_manipulator.slot_update_all_operation_limits)
        self.apply_seq_operations.connect(self.image_manipulator.slot_apply_seq_operations)
        self.apply_operation.connect(self.image_manipulator.slot_apply_operation)
        self.update_active_image.connect(self.image_manipulator.slot_update_active_image)
        self.reset_operations.connect(self.image_manipulator.slot_reset_operations)
        self.save_workflow_json.connect(self.image_manipulator.save_json)
        self.add_workflow_from_json.connect(self.image_manipulator.slot_add_workflow_from_json_file)
        self.load_workflow_json.connect(self.image_manipulator.slot_load_workflow_json_file)
        self.replicate_active_operation.connect(self.image_manipulator.slot_replicate_active_operation)

        self.image_manipulator.image_updated.connect(self.imageArea.set_image_slot)
        self.image_manipulator.reset_image_scale.connect(self.imageArea.normal_size_slot)
        self.image_manipulator.operations_updated.connect(self.update_view_list)
        self.image_manipulator.operations_updated.connect(self.update_param_list)

        self.select_view_widget.select_view.connect(self.image_manipulator.slot_update_active_image)
        self.select_view_widget.start_operation.connect(self.image_manipulator.slot_apply_operation)
        self.select_view_widget.start_recur_operation.connect(self.image_manipulator.slot_apply_recur_operation)
        self.update_view_list.connect(lambda: self.select_view_widget.refresh_view_list(self.image_manipulator)) # should be a slot?
        self.imageArea.mouseClickSignal.connect(self.parameter_widget.point_click_event)

        self.parameter_widget.adjust_parameter.connect(self.image_manipulator.slot_update_parameter)
        self.parameter_widget.rename_operation.connect(self.image_manipulator.slot_rename_operation)
        self.parameter_widget.adjust_input_operation.connect(self.image_manipulator.slot_update_input_image)
        self.parameter_widget.update_operation.connect(self.image_manipulator.slot_update_operation_limits)
        self.add_operation_page.connect(self.parameter_widget.add_operation_page)
        self.update_param_list.connect(self.parameter_widget.refresh_operation_pages)
        self.select_view_widget.select_param_view.connect(self.parameter_widget.update_param_view)

        self.select_view_widget.delete_operation.connect(self.image_manipulator.slot_remove_operation)
        self.select_view_widget.delete_operation.connect(self.parameter_widget.update_param_view)
        self.select_view_widget.delete_operation.connect(lambda: self.select_view_widget.refresh_view_list(self.image_manipulator))

        # self.select_view_widget.show()
        self.select_view_dock_widget = QDockWidget(self)
        self.select_view_dock_widget.setWindowTitle("Select View")
        self.select_view_dock_widget.setWidget(self.select_view_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,self.select_view_dock_widget)

        self.parameter_dock_widget = QDockWidget(self)
        self.parameter_dock_widget.setWindowTitle("Parameters")
        self.parameter_dock_widget.setWidget(self.parameter_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.parameter_dock_widget)
       
        self.image_thread.start()
        self.createActions()

        self.status_label = QLabel("Open Image")
        self.statusBar().addWidget(self.status_label)
        self.info_label = QLabel()
        self.statusBar().addWidget(self.info_label)

        self.imageArea.mouseCoordSignal.connect(self.statusbar_update)
        self.image_manipulator.image_updated.connect(lambda img: self.info_label.setText(f'W:{img.width()},H:{img.height()}'))
        self.setCentralWidget(self.centreWidget)
        #self.resize(QGuiApplication.primaryScreen().availableSize() * 3 / 5)
        self.resize(1200,800)

    @Slot()
    def update_status_bar(self, status_text):
        self.status_label.setText(status_text)
    
    @Slot()
    def statusbar_update(self, text: str):
        self.status_label.setText(text)
    
    def moveEvent(self, event):
        super().moveEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
    
    def arrangeWindows(self):
        WINDOW_PADDING = 5
        self.select_view_widget.move(self.x()-self.select_view_widget.width()-WINDOW_PADDING,self.y())
        self.parameter_widget.move(self.x()+self.width()+WINDOW_PADDING,self.y())
        self.select_view_widget.raise_()
        self.parameter_widget.raise_()

    def closeEvent(self, event):
        self.select_view_widget.close()
        self.parameter_widget.close()
        
        self.image_thread.requestInterruption()
        self.image_thread.quit()
        self.image_thread.wait()

    def initializeImageFileDialog(self, dialog: QFileDialog, acceptMode: QFileDialog.AcceptMode, file_name = None):
        dialog.setDirectory(QDir.currentPath())

        if acceptMode == QFileDialog.AcceptMode.AcceptSave:
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setMimeTypeFilters(("image/tiff","image/jpeg","image/png","image/bmp"))
        elif acceptMode == QFileDialog.AcceptMode.AcceptOpen:
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            dialog.setMimeTypeFilters(("image/tiff","image/jpeg","image/png","image/bmp"))

        dialog.setAcceptMode(acceptMode)

    def openImageFile(self):
        dialog = QFileDialog(self, "Open File")
        self.initializeImageFileDialog(dialog, QFileDialog.AcceptMode.AcceptOpen)
        accepted = dialog.exec()
        if accepted:
            self.loadImageFile(dialog.selectedFiles()[0])

    def saveImageFile(self):
        dialog = QFileDialog(self, "Save File")
        self.initializeImageFileDialog(dialog, QFileDialog.AcceptMode.AcceptSave)
        accepted = dialog.exec()
        if accepted:
            self.save_image_file.emit(dialog.selectedFiles()[0])

    def loadImageFile(self, fileName):
        self.load_new_image.emit(fileName)
        self.stretchToWindowAct.setEnabled(True)
        self.updateActions()
        if (not self.stretchToWindowAct.isChecked()):
            self.imageArea.reset_image_size()

    def initializeJSONFileDialog(self, dialog: QFileDialog, acceptMode: QFileDialog.AcceptMode):
        dialog.setDirectory(QDir.currentPath())
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("JSON (*.json)")
        dialog.setAcceptMode(acceptMode)

    @Slot()
    def fileReadErrorSlot(self):
        QMessageBox.information(self, QGuiApplication.applicationDisplayName, "Cannot load file")

    def createActions(self):
        menu_bar = self.menuBar()
        fileMenu = menu_bar.addMenu("&File")

        self.openAct = fileMenu.addAction("&Open Image")
        self.openAct.setShortcut(QKeySequence.StandardKey.Open)
        self.openAct.triggered.connect(self.openImageFile)
        
        self.openCADAct = fileMenu.addAction("Open &CAD File")
        self.openCADAct.triggered.connect(lambda: create_operation(CADOperation))


        self.saveActiveImageAct = fileMenu.addAction("Save &Active Image")
        self.saveActiveImageAct.setShortcut("Ctrl+S")
        self.saveActiveImageAct.triggered.connect(self.saveImageFile)
                
        fileMenu.addSeparator()
        
        self.saveWorkflowAct = fileMenu.addAction("&Save Workflow")
        self.saveWorkflowAct.setShortcut("Ctrl+Shift+S")
        self.saveWorkflowAct.triggered.connect(self.saveJSON)

        self.loadWorkflowAct = fileMenu.addAction("&Load Workflow")
        self.loadWorkflowAct.setShortcut("Ctrl+Shift+O")
        self.loadWorkflowAct.triggered.connect(self.loadJSON)

        self.appendWorkflowAct = fileMenu.addAction("&Append Workflow")
        self.appendWorkflowAct.setShortcut("Ctrl+Shift+I")
        self.appendWorkflowAct.triggered.connect(lambda: self.loadJSON(appendFlag=True))

        self.resetOperationsAct = fileMenu.addAction("&Reset Operations")
        self.resetOperationsAct.triggered.connect(self.resetAllOperations)

        fileMenu.addSeparator()

        self.exitAct = fileMenu.addAction("E&xit")
        self.exitAct.setShortcut("Ctrl+Q")
        self.exitAct.triggered.connect(self.close)

        editMenu = menu_bar.addMenu("&Edit")

        self.runOperation = editMenu.addAction("Run active operation")
        self.runOperation.setShortcut("Ctrl+G")
        self.runOperation.triggered.connect(self.activateApplyCurrentOperation)

        self.seqOperations = editMenu.addAction("Run all operations (Sequential)")
        self.seqOperations.setShortcut("Ctrl+R")
        self.seqOperations.triggered.connect(self.applyOperationsSequential)

        self.updateParam = editMenu.addAction("Update parameter limits")
        self.updateParam.setShortcut("Ctrl+U")
        self.updateParam.triggered.connect(self.update_all_operation_limits)

        self.testerAct = editMenu.addAction("Run Test Function")
        self.testerAct.triggered.connect(self.tester)

        editMenu.addSeparator()
        self.replicateAction = editMenu.addAction("Replicate current operation")
        self.replicateAction.triggered.connect(self.replicate_active_operation)

        viewMenu = menu_bar.addMenu("&View")

        self.zoomInAct = viewMenu.addAction("Zoom &In (25%)") 
        self.zoomInAct.setShortcuts((QKeySequence.StandardKey.ZoomIn, "Ctrl+="))
        # self.zoomInAct.setEnabled(False)
        self.zoomInAct.triggered.connect(self.imageArea.zoom_in_slot)

        self.zoomOutAct = viewMenu.addAction("Zoom &Out (25%)")
        self.zoomOutAct.setShortcut(QKeySequence.StandardKey.ZoomOut)
        # self.zoomOutAct.setEnabled(False)
        self.zoomOutAct.triggered.connect(self.imageArea.zoom_out_slot)

        self.normalSizeAct = viewMenu.addAction("&Reset Zoom")
        # self.normalSizeAct.setEnabled(False)
        self.normalSizeAct.triggered.connect(self.imageArea.normal_size_slot)
        self.normalSizeAct.setShortcut("Ctrl + 0")

        viewMenu.addSeparator()

        self.fitImgtoWindowAct = viewMenu.addAction("Fit Image to Window")
        self.fitImgtoWindowAct.triggered.connect(self.imageArea.ratio_fit_to_window)
        self.fitImgtoWindowAct.setShortcut("Ctrl+F")

        self.fitWindowToImgAct = viewMenu.addAction("Fit Window to Image")
        self.fitWindowToImgAct.triggered.connect(self.fitWindowToImage)

        self.bestFitAct = viewMenu.addAction("Best Fit Image")
        self.bestFitAct.triggered.connect(self.imageArea.ratio_fit_to_window)
        self.bestFitAct.triggered.connect(self.fitWindowToImage)
        self.bestFitAct.setShortcut("Ctrl+Shift+F")

        self.stretchToWindowAct = viewMenu.addAction("&Stretch to Window")
        # self.fitToWindowAct.setEnabled(False)
        self.stretchToWindowAct.setCheckable(True)
        self.stretchToWindowAct.triggered.connect(self.imageArea.stretch_to_window_slot)

        viewMenu.addSeparator()

        self.showViewWindowAct = viewMenu.addAction("Show View Selector Window")
        self.showViewWindowAct.triggered.connect(self.select_view_dock_widget.show)
        self.showParamWindowAct = viewMenu.addAction("Show Parameter Window")
        self.showParamWindowAct.triggered.connect(self.parameter_widget.show)

        CVMenu = menu_bar.addMenu("Operations")
        
        def add_CV_act(image_operation_class, menu, *args, **kwargs):
            action = menu.addAction(image_operation_class.__name__)
            action.triggered.connect(lambda: create_operation(image_operation_class, *args, **kwargs))
            return action

        def create_operation(image_operation_class: ImageOperation, *args, **kwargs):
            image_operation = image_operation_class(*args, **kwargs)
            self.add_operation.emit(image_operation)
            self.set_latest_input_image.emit(image_operation.id)
            self.update_operation_limits.emit(image_operation.id)
            self.update_view_list.emit()

        # New operations here
        # self.CVOperationLibrary = [RectangleOperation, GaussOperation, ThresholdOperation, AdaptiveThresholdOperation, RotateOperation, MorphologyOperation, \
        #                             GrayscaleOperation, InvertOperation, CompareOperation, ContourOperation, PolyContourOperation, TransferContourOperation,\
        #                                 CornerOperation, EuclidTransformOperation ]
        
        self.CVOperationLibrary = ImageOperation.__subclasses__()
        self.CVOperationAct_dict = {}
        self.CVOperation_category_menus = {}
        for cv_operation in self.CVOperationLibrary:
            if cv_operation.operation_category is None:
                op_category = "Uncategorized"
            else:
                op_category = str.capitalize(cv_operation.operation_category)
            if op_category not in self.CVOperation_category_menus:
                self.CVOperation_category_menus[op_category] = CVMenu.addMenu(op_category)
            self.CVOperationAct_dict[cv_operation.__class__.__name__] = add_CV_act(cv_operation,self.CVOperation_category_menus[op_category])

        helpMenu = menu_bar.addMenu("&Help")
        helpMenu.addAction("&About").triggered.connect(self.about)
    
    @Slot()
    def fitToWindow(self, isChecked):
        self.imageArea.stretch_to_window_slot(isChecked)
        # self.updateActions()

    @Slot()
    def fitWindowToImage(self):
        self.imageArea.setMinimumSize(self.imageArea.widget().size().grownBy(QMargins(1,1,1,1))) # Not sure why imageArea needs to be readjusted but it doesn't work properly otherwise
        self.centreWidget.resize(self.imageArea.widget().size().grownBy(QMargins(1,1,1,1))) # Prevent scrollbars from appearing by slight margins
        self.adjustSize()
        self.imageArea.setMinimumSize(QSize(0,0)) # The minimum width of the docked right window interferes since the adjust size doesn't always finish before the minimum size is reset to 0, or something to do with maximum size of window. Not sure

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.stretchToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.stretchToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.stretchToWindowAct.isChecked())
    
    @Slot()
    def about(self):
        QMessageBox.about(self,"Hello World! \n This is used to display an image.")

    @Slot()
    def saveJSON(self):
        dialog = QFileDialog(self, "Save File")
        self.initializeJSONFileDialog(dialog, acceptMode=QFileDialog.AcceptMode.AcceptSave)
        accepted = dialog.exec()
        if accepted:
            self.save_workflow_json.emit(dialog.selectedFiles()[0])

    @Slot()
    def loadJSON(self, appendFlag = False):
        dialog = QFileDialog(self, "Load File")
        self.initializeJSONFileDialog(dialog, acceptMode=QFileDialog.AcceptMode.AcceptOpen)
        accepted = dialog.exec()
        if accepted:
            json_file_name = dialog.selectedFiles()[0]
            if appendFlag:
                self.add_workflow_from_json.emit(json_file_name)
            else:
                self.load_workflow_json.emit(json_file_name)
        
    @Slot()
    def resetAllOperations(self):
        dialog_response = QMessageBox.question(self, "Reset all operations", "Are you sure you want to delete all operations?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if dialog_response == QMessageBox.StandardButton.Yes:
            self.reset_operations.emit()
            self.update_view_list.emit()
            self.update_param_list.emit()
        
    @Slot()
    def applyOperationsSequential(self):
        self.apply_seq_operations.emit()

    @Slot()
    def activateApplyCurrentOperation(self):
        operation_id = self.parameter_widget.stack_select_combobox.currentText()
        self.update_active_image.emit(operation_id)
        self.apply_operation.emit(operation_id)

    @Slot()
    def tester(self):
        # Testing code
        self.update_param_list.emit()
        pass

class SelectViewWidget(QWidget):
    select_view = Signal(object) 
    select_param_view = Signal(object)
    delete_operation = Signal(object) 
    start_operation = Signal(object)
    start_recur_operation = Signal(object)
    # emit signal of format (view_name)

    def __init__(self, ImageManipulatorObj: QTImageManipulator) -> None:
        super().__init__()
        self.layout = QVBoxLayout()
        self.refresh_view_list(ImageManipulatorObj)
        # self.setMinimumWidth(200)

    @Slot()
    def refresh_view_list(self, ImageManipulatorObj: QTImageManipulator):
        QWidget().setLayout(self.layout)
        self.layout = QVBoxLayout()
        self.scrollLayout = QVBoxLayout()
        # Manually add base_image view
        view_button = QPushButton("Base Image")
        self.scrollLayout.addWidget(view_button)
        view_button.pressed.connect(lambda image_name="base_image": self.select_view.emit(image_name))
        # TODO display currently displayed image name somewhere
        for id, cv_operation in ImageManipulatorObj.cv_operations_dict.items():
            sublayout = QHBoxLayout()
            view_button = QPushButton(id)
            delete_button = QPushButton("Delete")
            apply_button = QPushButton("Apply")
            # apply_button.hide() # Only have recursive to save space
            apply_recur_button = QPushButton("Apply (Recursive)")
            apply_recur_button.hide() # Hide recursive to save space
            sublayout.addWidget(view_button, 5)
            sublayout.addWidget(delete_button, 1)
            sublayout.addWidget(apply_button, 1)
            sublayout.addWidget(apply_recur_button, 1)
            self.scrollLayout.addLayout(sublayout)
            view_button.pressed.connect(lambda image_name=cv_operation.output_image_name: self.select_view.emit(image_name))
            view_button.pressed.connect(lambda image_name=cv_operation.output_image_name: self.select_param_view.emit(image_name))
            delete_button.pressed.connect(lambda image_name=cv_operation.id: self.delete_operation.emit(image_name))
            apply_button.pressed.connect(lambda operation_id=cv_operation.id: self.start_operation.emit(operation_id))
            apply_button.pressed.connect(lambda image_name=cv_operation.output_image_name: self.select_view.emit(image_name))
            apply_button.pressed.connect(lambda image_name=cv_operation.output_image_name: self.select_param_view.emit(image_name))
            apply_recur_button.pressed.connect(lambda operation_id=cv_operation.id: self.start_recur_operation.emit(operation_id))
            apply_recur_button.pressed.connect(lambda image_name=cv_operation.output_image_name: self.select_view.emit(image_name))

        self.scrollLayout.addStretch(0)
        self.scrollWidget = QWidget()
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollLayout.setContentsMargins(0, 0, 0, 0)
        self.scrollWidget.setLayout(self.scrollLayout)
        self.scrollArea.setWidget(self.scrollWidget)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.addWidget(self.scrollArea)
        self.setLayout(self.layout)
        # self.update()
    
    # TODO: Make this remove and add row instead of refreshing each time

class ParameterAdjustWidget(QWidget):
    adjust_parameter = Signal(object)    # emit signal with tuple of format (operation_id, parameter_name, new_value)
    rename_operation = Signal(object)   # emit signal with tuple of format (operation_id, new_operation_id)
    adjust_input_operation = Signal(object)   # emit signal with tuple of format (operation_id, new_input_operation_id) 
    update_operation = Signal(object)
    def __init__(self, ImageManipulatorObj: QTImageManipulator) -> None:
        super().__init__()
        self.layout = QVBoxLayout()
        self.image_manipulator = ImageManipulatorObj
        self.initialize_operation_pages()
        self.setMinimumWidth(250)
        self.pointAssignPointer = None

    @staticmethod
    def isOdd_validator(control: QAbstractSlider | QAbstractSpinBox):
        def validator_func():
            # print(control.value())
            if control.value() % 2 == 0:
                control.setValue(control.value()+1)
        return validator_func

    @Slot()
    def point_click_event(self, parameter_reference):
        if self.pointAssignPointer:
            parameter_obj, operation_id = self.pointAssignPointer
            parameter_obj.param_value = parameter_reference
            self.refresh_single_op_page(operation_id)
        self.pointAssignPointer = None
        self.refresh_single_op_page

    def generate_slider(self, param_type, min, max, param_value, precision=0, param_validation=None):
        precision_factor = 10 ** precision
        scaled_min = min * precision_factor
        scaled_max = max * precision_factor

        layout = QHBoxLayout()

        slider = QSlider()
        slider.setMinimum(scaled_min)
        slider.setMaximum(scaled_max)
        slider.setTracking(True)
        slider.setOrientation(Qt.Orientation.Horizontal)

        if param_type == 'int' or param_type == 'point':
            slider_alt = QSpinBox()
            slider.setValue(param_value)
            slider_alt.setValue(param_value)
            if param_validation == 'isOdd':
                slider.setSingleStep(2)
                slider_alt.setSingleStep(2)
            else:
                slider_alt.setSingleStep(1)
        elif param_type == 'float':
            slider_alt = QDoubleSpinBox()
            slider_alt.setDecimals(precision)
            slider_alt.setSingleStep(1/precision_factor)
            slider_alt.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
            slider.setValue(param_value*precision_factor)
            slider_alt.setValue(param_value)

        slider_alt.setMinimum(min)
        slider_alt.setMaximum(max)
        slider_alt.setValue(param_value)

        slider.setPageStep(np.maximum(round((scaled_max-scaled_min)/20,precision),1)) #around 20 steps, round depending on scale
        
        if param_validation == 'isOdd':
            slider.valueChanged.connect(ParameterAdjustWidget.isOdd_validator(slider))
            slider_alt.editingFinished.connect(ParameterAdjustWidget.isOdd_validator(slider_alt))

        if param_type == 'int':
            slider.valueChanged.connect(lambda: slider_alt.setValue(slider.value()))
            slider_alt.editingFinished.connect(lambda: slider.setValue(slider_alt.value()))
        elif param_type == 'float': #only allows int, need to modify to make it work for floats
            slider.valueChanged.connect(lambda: slider_alt.setValue(slider.value()/precision_factor))
            slider_alt.editingFinished.connect(lambda: slider.setValue(slider_alt.value()*precision_factor))

        layout.addWidget(slider_alt,1)
        layout.addWidget(slider,7)
        return layout, slider
    
    def parameter_control(self, operation_id, parameter_name, parameter_obj):
        if parameter_obj.param_type == 'img_file' or parameter_obj.param_type == 'cad_file':
            layout = QVBoxLayout()
            button = QPushButton("Choose file...")
            label = QLabel()
            label.setText(parameter_obj.param_value)
            def run_file_dialog(file_type):
                dialog = QFileDialog(self, "Open File")
                dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
                if file_type == 'img_file':
                    dialog.setMimeTypeFilters(("image/tiff","image/jpeg","image/png","image/bmp"))
                elif file_type == 'cad_file':
                    dialog.setNameFilters(("CAD (*.stl *.obj *.3mf)", "Any File (*)"))
                accepted = dialog.exec()
                if accepted:
                    file_path = dialog.selectedFiles()[0]
                    rel_file_path = os.path.relpath(file_path)
                    label.setText(rel_file_path) #Weird closure, does work
                    return rel_file_path
                else:
                    return ""
                
            button.released.connect(lambda file_type=parameter_obj.param_type: self.adjust_parameter.emit((operation_id, parameter_name, run_file_dialog(file_type=file_type))))
            layout.addWidget(button)
            layout.addWidget(label)
            return layout

        if parameter_obj.param_type == 'enum':
            # TODO should make this work with tuples of enum as well ideally
            layout = QFormLayout()
            label = QLabel(parameter_obj.param_label[0])
            combobox = QComboBox()
            enum_name_list = [item[0] for item in parameter_obj.param_enum]
            enum_value_list = [item[1] for item in parameter_obj.param_enum]
            combobox.addItems(enum_name_list)
            item_index = enum_value_list.index(parameter_obj.param_value)
            combobox.setCurrentIndex(item_index)
            signal_emit_func = lambda enum_index: self.adjust_parameter.emit((operation_id, parameter_name, parameter_obj.param_enum[enum_index][1]))
            combobox.activated.connect(signal_emit_func)
            layout.addRow(label, combobox)
            return layout
            
        if parameter_obj.param_type == "point": # need param_count=2
            layout = QFormLayout()
            control_elements = []
            for i in range(parameter_obj.param_count):  
                control_elements.append((*self.generate_slider('int',parameter_obj.param_min[i],parameter_obj.param_max[i],parameter_obj.param_value[i], precision=0, param_validation=parameter_obj.param_validation), QLabel(parameter_obj.param_label[i])))
            signal_emit_func = (lambda: self.adjust_parameter.emit((operation_id, parameter_name, [element.value() for element_layout, element, *null in control_elements])))
            for element_layout, element, label in control_elements:
                element.valueChanged.connect(signal_emit_func)
                layout.addRow(label,element_layout)
            point_button = QPushButton("Select point on image...")
            def assign_point():
                p_obj = parameter_obj
                o_id = operation_id
                self.pointAssignPointer = (parameter_obj, operation_id)
            point_button.pressed.connect(assign_point)
            layout.addRow(point_button)
            return layout
            
        # int or float
        precision = parameter_obj.precision

        layout = QFormLayout()
        control_elements = []
        if parameter_obj.param_count > 1:
            for i in range(parameter_obj.param_count):  
                control_elements.append((*self.generate_slider(parameter_obj.param_type,parameter_obj.param_min[i],parameter_obj.param_max[i],parameter_obj.param_value[i], precision, parameter_obj.param_validation), QLabel(parameter_obj.param_label[i])))
        else:
            control_elements.append((*self.generate_slider(parameter_obj.param_type,parameter_obj.param_min,parameter_obj.param_max,parameter_obj.param_value, precision, parameter_obj.param_validation), QLabel(parameter_obj.param_label)))

        signal_emit_func = None
        if parameter_obj.param_count == 1:
            if parameter_obj.param_type == 'float':
                signal_emit_func = lambda: self.adjust_parameter.emit((operation_id, parameter_name, control_elements[0][1].value()/(10**precision)))
            else: #int
                signal_emit_func = lambda: self.adjust_parameter.emit((operation_id, parameter_name, control_elements[0][1].value()))
        else:
            if parameter_obj.param_type == 'float':
                signal_emit_func = (lambda: self.adjust_parameter.emit((operation_id, parameter_name, [element.value()/(10**precision) for element_layout, element, *null in control_elements])))
            else: #int
                signal_emit_func = (lambda: self.adjust_parameter.emit((operation_id, parameter_name, [element.value() for element_layout, element, *null in control_elements])))

        for element_layout, element, label in control_elements:
            # Rewrite this to connect signal directly to slot in image_manipulator? No, leave connections in MainWindow
            element.valueChanged.connect(signal_emit_func)
            layout.addRow(label,element_layout)

        return layout
    
    def initialize_operation_pages(self):
        self.layout = QVBoxLayout()
        self.refresh_operation_pages()
    
    def refresh_operation_pages(self):
        # Refresh layout by assigning to temporary QWidget
        QWidget().setLayout(self.layout)
        self.layout = QVBoxLayout()
        # Combobox to select page in stacked pages, one per operation
        self.stack_select_combobox = QComboBox()
        #self.stack_select_combobox.addItems(self.image_manipulator.cv_operations_dict.keys())
        self.layout.addWidget(self.stack_select_combobox)
        self.stack_widget = QStackedWidget()
        self.stack_select_combobox.activated.connect(self.stack_widget.setCurrentIndex)
        # Generate set of controls for each operation in dictionary
        self.operation_widgets = {}
        # Generate blank base_image param page
        base_image = "base_image"
        self.stack_select_combobox.addItem("base_image")
        base_image_widget = QWidget()
        self.stack_widget.addWidget(base_image_widget)
        self.operation_widgets[base_image] = base_image_widget
        for operation_id, cv_operation in self.image_manipulator.cv_operations_dict.items():
            self.add_operation_page(operation_id)
        self.stack_select_combobox.setCurrentText(self.image_manipulator.active_image_name)
        self.stack_widget.setCurrentWidget(self.operation_widgets[self.image_manipulator.active_image_name])
        self.layout.addWidget(self.stack_widget)
        self.setLayout(self.layout)
        self.update()

    def refresh_single_op_page(self, operation_id):
        op_widget = self.operation_widgets[operation_id]
        page_widget = self.create_operation_widget(operation_id)

        widget_index = self.stack_widget.indexOf(self.operation_widgets[operation_id])
        self.stack_widget.removeWidget(self.operation_widgets[operation_id])
        self.operation_widgets[operation_id].setParent(QWidget())  #Deletes widget in theory

        self.stack_widget.insertWidget(widget_index, page_widget)
        self.stack_widget.setCurrentIndex(widget_index)
        self.operation_widgets[operation_id]=page_widget
        self.update()

    def create_operation_widget(self, operation_id):
        layout = QVBoxLayout()
        cv_operation = self.image_manipulator.cv_operations_dict[operation_id]
        # Renaming Operation
        editId_widget = QLineEdit()
        editId_widget.setText(cv_operation.id)
        
        @Slot()
        def check_rename():
            cv_op = cv_operation
            new_id = editId_widget.text()
            if cv_op.id == new_id:
                #Conveniently fixes a QT bug that will double fire the event if enter is pressed and contents not changed, and also prevents unnessecary renaming
                return
            if new_id == "":
                new_id = "Untitled_Operation"
            if (new_id in self.image_manipulator.cv_operations_dict):
                editId_widget.setText(cv_op.id)
                return
            editId_widget.setText(new_id)
            self.rename_operation.emit((cv_operation.id, new_id))

        editId_widget.editingFinished.connect(check_rename)
        # editId_widget.setValidator(self.rename_validator)
        layout.addWidget(editId_widget)

        layout.addWidget(QLabel(f'{cv_operation.__class__.__name__}:'))
        # Generate control for each param
        for param_name, parameter_obj in cv_operation.OperationParameter_dict.items():
            layout.addWidget(QLabel(f'{param_name}:'))
            #print(parameter_obj.param_type)
            param_control = self.parameter_control(operation_id, param_name, parameter_obj)
            layout.addLayout(param_control)

        # Input Operation Selection
        # Cyclical input loops, okay if each operation is applied manually, if operations can be applied in sequence, need to ensure that input of operation must be earlier in the hierarchy
        # Currently doing this by preventing selecting operations as input if created later, (not updating the combobox)
        # TODO But could instead check at apply time by making sure no loops (ie no repeat operations are found) when following up the input links
        sublayout_input_select = QFormLayout()
        if cv_operation.input_image_count == 1:
            input_combobox = QComboBox()
            input_combobox.addItems([image_name for image_name in self.image_manipulator.cv_image_dict if image_name != cv_operation.output_image_name])
            input_combobox.setCurrentText(cv_operation.input_image_name)
            input_combobox.currentTextChanged.connect(lambda input_image_name: self.adjust_input_operation.emit((operation_id,input_image_name)))
            # input_combobox.activated.connect(lambda: self.update_operation.emit(operation_id))
            # input_combobox.activated.connect(lambda: self.refresh_single_op_page(operation_id))
            sublayout_input_select.addRow("Input image:", input_combobox)
        else:
            input_combobox_list = []
            for i in range(cv_operation.input_image_count):
                input_combobox = QComboBox()
                input_combobox_list.append(input_combobox)
                input_combobox.addItems([image_name for image_name in self.image_manipulator.cv_image_dict if image_name != cv_operation.output_image_name])
                # TODO Update combobox behaivour to be consistent, all operations appear when loaded from JSON, only previous operations appear when added sequentially by controls
                input_combobox.setCurrentText(cv_operation.input_image_name[i])
                sublayout_input_select.addRow(f"Input image {i+1}:", input_combobox)
            for input_combobox in input_combobox_list:
                input_combobox.currentTextChanged.connect(lambda: self.adjust_input_operation.emit((operation_id,[in_combo.currentText() for in_combo in input_combobox_list])))
        layout.addStretch(0)
        layout.addLayout(sublayout_input_select)
        # Have a page for each operation
        page_widget = QWidget()
        page_widget.setLayout(layout)
        return page_widget
    
    def add_operation_page(self, operation_id):
        self.stack_select_combobox.addItem(operation_id)
        page_widget = self.create_operation_widget(operation_id)

        self.stack_widget.addWidget(page_widget)
        self.operation_widgets[operation_id]=page_widget
        self.update()
    
    @Slot()
    def remove_param_controls(self, operation_id):
        widget_index = self.stack_widget.indexOf(self.operation_widgets[operation_id])
        self.stack_widget.removeWidget(self.operation_widgets[operation_id])
        self.stack_select_combobox.removeItem(widget_index)

    @Slot()
    def update_param_view(self, image_name: str):
        if image_name == "base_image": #remove this when shifting base_image loading
            return
        self.stack_select_combobox.setCurrentText(image_name.removesuffix('_image'))
        self.stack_widget.setCurrentIndex(self.stack_select_combobox.currentIndex())
