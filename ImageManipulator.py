import json
import inspect
import datetime
import uuid
import copy
import cv2
import numpy as np

class ImageManipulator():

    def __init__(self):
        super().__init__()
        self.active_image_name = 'base_image'
        self.cv_image_dict = {'base_image': None}
        self.cv_data_dict = {'base_image': {}}
        self.cv_operations_dict = {}

    def load_image(self, file_name: str):
        new_image = cv2.imread(file_name,cv2.IMREAD_COLOR)
        if new_image is None:
            raise(Exception("Image file does not exist"))
        self.cv_image_dict['base_image'] = new_image
        self.reset_active_image()
        self.update_all_operation_limits()

    def save_active_results(self, file_name:str):
        self.save_image_data(self.active_image_name,file_name)

    def save_image_data(self, image_name, file_name):
        # this assumes file has image extensions
        self.save_image(image_name, file_name)
        self.save_data(image_name, f'{file_name.split('.')[0]}_data.json')

    def save_image(self, image_name, file_name):
        cv2.imwrite(file_name,self.cv_image_dict[image_name])

    def save_data(self, image_name, file_name):
        cv_data = {}
        for key, data in self.cv_data_dict[image_name].items():
            if isinstance(data,np.ndarray):
                cv_data[key] = data.tolist()
            elif isinstance(data,(dict,list,tuple)):
                cv_data[key] = data
            elif isinstance(data, uuid.UUID):
                cv_data[key] = str(data)
            else:
                cv_data[key] = data
        json_data = json.dumps(cv_data)
        
        with open(file_name,'w+') as data_file:
            data_file.write(json_data)

    def set_base_image(self, img : np.ndarray):
        self.cv_image_dict['base_image'] = img

    def reset_active_image(self):
        self.active_image_name = 'base_image'
    
    def update_active_image(self, view_name):
        self.active_image_name = view_name
    
    def add_operation(self, image_operation): # Adds supplied operation object to dict and creates corresponding blank output image in dict
        self.cv_operations_dict[image_operation.id] = image_operation
        self.cv_image_dict[image_operation.output_image_name] = None
        self.cv_data_dict[image_operation.output_image_name] = {}
        return image_operation
    
    def remove_operation(self, operation_id):
        del self.cv_operations_dict[operation_id]
        del self.cv_image_dict[operation_id]
        del self.cv_data_dict[operation_id]

    def replicate_operation(self, operation_id):
        # Terrible JSON hack
        operation_obj = self.cv_operations_dict[operation_id]
        new_op = copy.deepcopy(operation_obj)
        new_op.id = f'{operation_id}_copy'
        new_op.output_image_name = new_op.id
        self.add_operation(new_op)

    def replicate_active_operation(self):
        self.replicate_operation(self.active_image_name)

    def reset_operations(self):
        backup_base_image = self.cv_image_dict['base_image']
        self.active_image_name = 'base_image'
        self.cv_image_dict = {'base_image': backup_base_image}
        self.cv_data_dict = {'base_image': {}}
        self.cv_operations_dict = {}

    def update_parameter(self, operation_id, parameter_name, parameter_value):
        operation_obj = self.cv_operations_dict[operation_id]
        operation_obj.set_parameter(parameter_name,parameter_value)
        #self.apply_operation(operation_id)
    
    def update_operation_limits(self, operation_id):
        operation_obj = self.cv_operations_dict[operation_id]
        operation_obj.update_param_limits(self.cv_image_dict)

    def update_all_operation_limits(self):
        for operation in self.cv_operations_dict.values():
            operation.update_param_limits(self.cv_image_dict)
    
    def update_input_image(self, operation_id, input_image_name):
        operation_obj = self.cv_operations_dict[operation_id]
        operation_obj.set_input_image(input_image_name)

    def set_input_image_to_latest(self, operation_id):
        operation_obj = self.cv_operations_dict[operation_id]
        if operation_obj.input_image_count == 1:
            operation_obj.set_input_image(list(self.cv_image_dict.keys())[-2]) #Latest will be the current operation
        else:
            operation_obj.set_input_image([list(self.cv_image_dict.keys())[-2] for i in range(operation_obj.input_image_count)]) 

    def rename_operation(self, old_id, new_id):
        operation_obj = self.cv_operations_dict.pop(old_id)
        self.cv_operations_dict[new_id] = operation_obj
        self.cv_image_dict[new_id] = self.cv_image_dict.pop(old_id)
        self.cv_data_dict[new_id] = self.cv_data_dict.pop(old_id)
        # Will shuffle down to bottom as if new operation
        operation_obj.id = new_id
        operation_obj.output_image_name = operation_obj.id

        for operation in self.cv_operations_dict.values():
            if operation.input_image_count == 1:
                if operation.input_image_name == old_id:
                    operation.input_image_name = new_id
            else:
                indices = [i for i, val in enumerate(operation.input_image_name) if val == old_id]
                for index in indices:
                    operation.input_image_name[index] = new_id
        
        if self.active_image_name == old_id:
            self.active_image_name = new_id

    def apply_operation(self, operation_id):
        self.cv_operations_dict[operation_id].process_operation(self.cv_image_dict, self.cv_data_dict)

    def apply_recur_operation(self, operation_id):
        cv_operation = self.cv_operations_dict[operation_id]
        if cv_operation.input_image_count > 1:
            return # TODO Support multi-input operations recursively
        # Applies all pre-requisite operations in chain
        input_image = cv_operation.input_image_name
        if input_image != 'base_image':
            self.apply_recur_operation(input_image)
        cv_operation.process_operation(self.cv_image_dict, self.cv_data_dict)
        # Non recursive: while loop to trace up the input images until base_image, then for loop over all images in order
    
    def apply_seq_operations(self):
        # Applies all operations in dict sequentially, temporary function
        for cv_operation in self.cv_operations_dict.values():
            cv_operation.process_operation(self.cv_image_dict, self.cv_data_dict)

    def save_json(self, json_file_name):
        with open(json_file_name,'w+') as json_file:
            json_file.write(self.workflow_to_json())
            
    def workflow_to_json(self):
        operations_json_dump = []
        for cv_operation in self.cv_operations_dict.values():
            operations_json_dump.append(cv_operation.to_json())
        json_dump = f'{{"Workflow_creation_date": "{datetime.datetime.now()}", "operations":[{','.join(operations_json_dump)}]}}'
        return json_dump
    
    def load_workflow_json_file(self, json_file_name):
        # reset dict
        self.reset_operations()
        self.add_workflow_from_json_file(json_file_name, bypassUUID = True)
        
    def add_workflow_from_json_file(self, json_file_name, bypassUUID = False):
        # Parse JSON
        with open(json_file_name,'r') as json_file:
            json_object = json.load(json_file)
        json_operations = json_object['operations']
        # Lookup dictionary (might be better to make this a global)
        import ImageOperations
        operation_tuples = inspect.getmembers(ImageOperations, inspect.isclass) 
        operation_lookup_dict = {class_name: class_object for class_name, class_object in operation_tuples} 
        for json_operation in json_operations:
            if json_operation['operation_class'] in operation_lookup_dict:
                new_operation_object = operation_lookup_dict[json_operation['operation_class']](input_image=json_operation['input_image'],input_image_count=json_operation['input_image_count'],id=json_operation['operation_id'], bypassUUID=bypassUUID)
                for param_key, param_value in json_operation['operation_params'].items():
                    if param_key in new_operation_object.OperationParameter_dict:
                        new_operation_object.OperationParameter_dict[param_key].param_value = param_value
                        print(f'Loading JSON: {json_operation['operation_id']}:{param_key}:{param_value}')
                    else:
                        print(f"Warning: JSON has unsupported parameter {json_operation['operation_class']}:{param_key}")
            else:
                print(f"Warning: JSON has unsupported operation {json_operation['operation_class']}")

            self.add_operation(new_operation_object)

        self.update_all_operation_limits()