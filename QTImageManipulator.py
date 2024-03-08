from PySide6.QtCore import QObject, Qt, Signal, Slot, QStandardPaths, QDir, QThread, QEvent
from PySide6.QtGui import QImage
from ImageManipulator import ImageManipulator
import cv2

class QTImageManipulator(QObject, ImageManipulator):
    image_updated = Signal(QImage)
    operations_updated = Signal()
    reset_image_scale = Signal()
    file_read_fail = Signal()

    @Slot()
    def slot_load_image(self, file_name: str):
        return super().load_image(file_name)
    
    @Slot() 
    def slot_save_active_results(self, file_name: str):
        return super().save_active_results(file_name)

    def update_qt_image(self):
        if self.active_image_name not in self.cv_image_dict:
            return
        if self.cv_image_dict[self.active_image_name] is None:
            return
        color_cv_image = cv2.cvtColor(self.cv_image_dict[self.active_image_name], cv2.COLOR_BGR2RGB)
        qt_image = QImage(color_cv_image.data, color_cv_image.shape[1], color_cv_image.shape[0], color_cv_image.shape[1]*color_cv_image.shape[2], QImage.Format_RGB888)
        self.image_updated.emit(qt_image)
    
    @Slot()
    def slot_update_active_image(self, view_name):
        super().update_active_image(view_name)
        self.update_qt_image()

    @Slot()
    def slot_reset_active_image(self):
        super().reset_active_image()
        self.update_qt_image()
        self.reset_image_scale.emit()

    @Slot()
    def slot_reset_operations(self):
        super().reset_operations()
        self.operations_updated.emit()
    
    @Slot()
    def slot_add_operation(self, image_operation):
        super().add_operation(image_operation)
        self.operations_updated.emit()
    
    @Slot()
    def slot_remove_operation(self, image_operation):
        super().remove_operation(image_operation)
        self.operations_updated.emit()
    
    @Slot()
    def slot_replicate_active_operation(self):
        super().replicate_active_operation()
        self.operations_updated.emit()

    @Slot()
    def slot_update_parameter(self, signal_obj):
        return super().update_parameter(*signal_obj)
    
    @Slot()
    def slot_update_operation_limits(self, operation_id):
        return super().update_operation_limits(operation_id)
    
    @Slot()
    def slot_update_all_operation_limits(self):
        super().update_all_operation_limits()
        self.operations_updated.emit()
    
    @Slot()
    def slot_update_input_image(self, signal_obj):
        super().update_input_image(*signal_obj)
        # self.operations_updated.emit()
    
    @Slot()
    def slot_rename_operation(self, signal_obj):
        super().rename_operation(*signal_obj)
        self.operations_updated.emit()

    @Slot()
    def slot_apply_operation(self, operation_id):
        super().apply_operation(operation_id)
        self.update_qt_image()

    @Slot()
    def slot_apply_active_operation(self):
        super().apply_active_operation()
        self.update_qt_image()

    @Slot()
    def slot_apply_recur_operation(self, operation_id):
        super().apply_recur_operation(operation_id)
        self.update_qt_image()

    @Slot()
    def slot_apply_seq_operations(self):
        super().apply_seq_operations()
        self.update_qt_image()

    @Slot()
    def slot_load_workflow_json_file(self, json_file):
        super().load_workflow_json_file(json_file)
        self.operations_updated.emit()
    
    @Slot()
    def slot_add_workflow_from_json_file(self, json_file, bypassUUID=False):
        super().add_workflow_from_json_file(json_file, bypassUUID)
        print("load json")
        self.operations_updated.emit()
