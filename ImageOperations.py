import cv2
import numpy as np
import json
import uuid
import copy
import trimesh
import skimage

class OperationParameter(object):
    # for numeric controls
    def __init__(self, param_type = 'int', param_count = 1, param_min = 0, param_max = 100, param_value=None, param_label=None, param_validation = None, param_enum : list = [], float_precision = None) -> None:
        # UPDATE param_count for all P definitions, depreciate tuple and rely on param_count instead
        if param_type !='cad_file' and param_type != 'img_file' and param_value is None:
            param_value = param_min
        if param_label is None:
            if param_count > 1:
                param_label = [param_type for i in range(param_count)]
            else:
                param_label = param_type

        self.param_type = param_type
        self.param_count = param_count
        self.param_validation = param_validation
        self.param_label = param_label
        self.param_enum = param_enum
        self.param_min = param_min
        self.param_max = param_max
        self.param_value = param_value

        if float_precision is None:
            if self.param_type == 'int':
                self.precision = 0
            elif self.param_type == 'float':
                self.precision = 3
            else:
                self.precision = 0
        else:
            self.precision = float_precision

    def copy(self):
        return self.__copy__()
    
    def __copy__(self):
        return OperationParameter(self.param_type, self.param_count, self.param_min, self.param_max, self.param_value, self.param_label, self.param_validation, self.param_enum)
    
    # def __deepcopy__(self,memo):
    #     return self.__copy__()


def isOdd(number):
    return number % 2 == 1

class IdCounter(type):
    def __new__(cls, *args, **kwargs):
        new_class = super().__new__(cls, *args, **kwargs)
        new_class.id_count = 0
        return new_class

class ImageOperation():
    operation_category = None
    def __init__(self, input_image="base_image", input_image_count=1, id=None, bypassUUID=False) -> None:
        if id is None:
            self.id = str(uuid.uuid4())
        elif bypassUUID:
            self.id = id
        else:
            self.id = id

        self.input_image_name = input_image
        self.input_image_count = input_image_count
        self.output_image_name = self.id
        self.cv_function = None
        self.is_in_place = False
        self.has_retval = False
        self.OperationParameter_dict = {}

    def set_input_image(self, input_image_name):
        self.input_image_name = input_image_name

    def set_parameter(self, parameter_name, parameter_value):
        if parameter_name in self.OperationParameter_dict:
            self.OperationParameter_dict[parameter_name].param_value = parameter_value
            #print(f'{parameter_name} set to {parameter_value}')
        else:
            raise KeyError("Parameter name not found")
        
    def set_param_limits(self, parameter_name, param_min = None, param_max = None, param_value = None):
        if parameter_name not in self.OperationParameter_dict:
            raise KeyError("Parameter name not found")
        
        if param_min is not None:
            self.OperationParameter_dict[parameter_name].param_min = param_min
        if param_max is not None:
            self.OperationParameter_dict[parameter_name].param_max = param_max
        if param_value is not None:
            self.OperationParameter_dict[parameter_name].param_value = param_value
        
    def update_param_limits(self, image_dict): #virtual
        pass

    def input_image_exists(self, image_dict):
        if image_dict[self.input_image_name] is None:
            return False
        return True

    def make_param_dict(self):
        return {param_name: control_dict.param_value for param_name, control_dict in self.OperationParameter_dict.items()}

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        if self.is_in_place:
            image_dict[self.output_image_name] = image_dict[self.input_image_name].copy()
            # In-place operation, need to make copy first
            self.cv_function(image_dict[self.output_image_name],**parameter_arguments, **kwargs)
        else:
            if self.has_retval:
                retval, image_dict[self.output_image_name] = self.cv_function(image_dict[self.input_image_name],**parameter_arguments, **kwargs)
            else:
                image_dict[self.output_image_name] = self.cv_function(image_dict[self.input_image_name],**parameter_arguments, **kwargs)

    def to_json(self):
        parameter_arguments = self.make_param_dict()
        json_dict = {"operation_class":self.__class__.__name__, "operation_id":self.id, "input_image_count": self.input_image_count,"input_image":self.input_image_name, "operation_params":parameter_arguments}
        json_dump = json.dumps(json_dict)
        return json_dump

class ImageLoadOperation(ImageOperation):
    operation_category = "Import"
    def __init__(self, input_image="null", *args, **kwargs) -> None:
        super().__init__(input_image, *args, **kwargs)
        self.OperationParameter_dict['file_name'] = OperationParameter('img_file',1,param_value="")
    
    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        image_dict[self.output_image_name] = cv2.imread(parameter_arguments['file_name'],cv2.IMREAD_COLOR)
        
class CADOperation(ImageOperation):
    operation_category = "Import"
    def __init__(self, cad_file_name:str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        point3d = OperationParameter('float',3,(0,0,0),(10,10,10))
        normal3d = OperationParameter('int',3,(-1,-1,-1),(1,1,1),(0,1,0))
        cadFile = OperationParameter('cad_file',1,param_value=cad_file_name)
        self.OperationParameter_dict = {"file_name": cadFile,"scale":OperationParameter('float',1,0.0,500,0.1,"px/mm",float_precision=3),'origin':point3d,'normal':normal3d}
        self.cad_mesh = None
        self.currentlyLoaded = None
        # Scale may impact size of image and stall thread
    
    def load_file(self):
        file_name = self.OperationParameter_dict['file_name'].param_value
        if self.OperationParameter_dict['file_name'].param_value is None:
            return
        else:
            self.cad_mesh = trimesh.load_mesh(file_name)
            self.currentlyLoaded = self.OperationParameter_dict['file_name'].param_value
        if self.cad_mesh is None:
            print("CAD File Not Found")

    def generate_slice(self, scale, normal=(0,1,0), origin=(0,0,0), file_name=None):
        if self.cad_mesh is None:
            if self.OperationParameter_dict['file_name'].param_value is None:
                return
            else:
                self.load_file()
        path3dim = self.cad_mesh.section(normal,origin)
        path2dim, to_3D = path3dim.to_planar(normal=normal)
        slice_image = np.array(path2dim.rasterize(scale).convert('RGB')) # PIL to OpenCV image array
        return slice_image
    
    def process_operation(self, image_dict, data_dict):
        if self.currentlyLoaded != self.OperationParameter_dict['file_name'].param_value:
            self.load_file()
        parameter_arguments = self.make_param_dict()
        # print(parameter_dict['scale'])
        parameter_arguments['scale'] = 1 / parameter_arguments['scale'] # convert from px/mm to mm/px
        raster_img = self.generate_slice(**parameter_arguments)
        image_dict[self.output_image_name] = cv2.cvtColor(raster_img, cv2.COLOR_RGB2GRAY)

    def update_param_limits(self, image_dict):
        if self.cad_mesh is None:
            if self.OperationParameter_dict['file_name'].param_value is None:
                return
            else:
                self.load_file()
        bounds = self.cad_mesh.bounds
        # point3d = OperationParameter('float',3,(0,0,0),(bounds[1,0],bounds[1,1],bounds[1,2]))
        self.set_param_limits('origin', param_max = (bounds[1,0],bounds[1,1],bounds[1,2]))

class RectangleOperation(ImageOperation):
    operation_category = "Draw"
    def __init__(self, input_image="base_image", *args, **kwargs) -> None:
        super().__init__(input_image, *args, **kwargs)
        self.cv_function = cv2.rectangle
        self.is_in_place = True
        coord_value_ui = OperationParameter('point',2,(0,0),(1200,1200),(0,0),param_label=('x','y'))
        self.OperationParameter_dict = {"pt1": coord_value_ui.copy(), "pt2": coord_value_ui.copy(), "thickness": OperationParameter('int',1,-1,200,3), "color": OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))}
    
    def update_param_limits(self, image_dict):
        if not self.input_image_exists(image_dict):
            return
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        self.set_param_limits('pt1', param_max=img_size)
        self.set_param_limits('pt2', param_max=img_size)

class GaussOperation(ImageOperation):
    operation_category = "Filter"
    def __init__(self, input_image="base_image", *args, **kwargs) -> None:
        super().__init__(input_image, *args, **kwargs)
        self.cv_function = cv2.GaussianBlur
        self.OperationParameter_dict = {'ksize': OperationParameter('int',2,(0,0),(105,105),(55,55),param_validation='isOdd')}
    
    def process_operation(self, image_dict, data_dict, **kwargs):
        return super().process_operation(image_dict, data_dict, sigmaX=0, sigmaY=0)

class ThresholdOperation(ImageOperation):
    operation_category = "Filter"
    def __init__(self, input_image="base_image", *args, **kwargs) -> None:
        super().__init__(input_image, *args, **kwargs)
        self.cv_function = cv2.threshold
        self.has_retval = True
        self.OperationParameter_dict = {'thresh': OperationParameter('int',1,0,255)}

    def process_operation(self, image_dict, data_dict, **kwargs):
        return super().process_operation(image_dict, data_dict, maxval = 255, type = cv2.THRESH_BINARY)

class RotateOperation(ImageOperation):
    operation_category = "Resize"
    def __init__(self, input_image="base_image", *args, **kwargs) -> None:
        super().__init__(input_image, *args, **kwargs)
        self.cv_function = cv2.rotate
        rotate_enum = (('90 Clockwise',cv2.ROTATE_90_CLOCKWISE), ('180 Clockwise', cv2.ROTATE_180), ('270 Clockwise', cv2.ROTATE_90_COUNTERCLOCKWISE))
        self.OperationParameter_dict = {'rotateCode': OperationParameter('enum',1,0,2,cv2.ROTATE_90_CLOCKWISE,('Rotation Type',),param_enum=rotate_enum)}

class GrayscaleOperation(ImageOperation):
    operation_category = "Filter"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_function = cv2.cvtColor
    def process_operation(self, image_dict, data_dict, **kwargs):
        return super().process_operation(image_dict, data_dict, code=cv2.COLOR_BGR2GRAY)

class InvertOperation(ImageOperation):
    operation_category = "Filter"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_function = cv2.bitwise_not

class BitwiseAndOperation(ImageOperation):
    operation_category = "Combine"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 2

    def process_operation(self, image_dict, data_dict, **kwargs):
        img1 = image_dict[self.input_image_name[0]]
        img2 = image_dict[self.input_image_name[1]]
        img1_channels = img1.shape[2] if len(img1.shape) > 2 else 1
        img2_channels = img2.shape[2] if len(img2.shape) > 2 else 1
        output_img = None
        if img1_channels != img2_channels:
            if img1_channels < img2_channels:
                output_img = cv2.bitwise_and(img2, np.dstack((img1, img1, img1)))
            else:
                output_img = cv2.bitwise_and(img1, np.dstack((img2, img2, img2)))
        else:
            output_img = cv2.bitwise_and(img1, img2)
        image_dict[self.output_image_name] = output_img
        
class BitwiseDifferenceOperation(ImageOperation):
    operation_category = "Combine"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 2

    def process_operation(self, image_dict, data_dict, **kwargs):
        img1 = cv2.bitwise_not(image_dict[self.input_image_name[0]])
        img2 = image_dict[self.input_image_name[1]]
        img1_channels = img1.shape[2] if len(img1.shape) > 2 else 1
        img2_channels = img2.shape[2] if len(img2.shape) > 2 else 1
        output_img = None
        if img1_channels != img2_channels:
            if img1_channels < img2_channels:
                output_img = cv2.bitwise_and(img2, np.dstack((img1, img1, img1)))
            else:
                output_img = cv2.bitwise_and(img1, np.dstack((img2, img2, img2)))
        else:
            output_img = cv2.bitwise_and(img1, img2)
        image_dict[self.output_image_name] = output_img
    
class CompareOperation(ImageOperation):
    operation_category = "Combine"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 2
        color_enum = (('Blue',0),('Green',1),('Red',2))
        color_ui_1 = OperationParameter('enum',1,0,2,0,"Image 1 Channel",param_enum=color_enum)
        color_ui_2 = OperationParameter('enum',1,0,2,1,"Image 2 Channel",param_enum=color_enum)
        self.OperationParameter_dict = {'img1ch': color_ui_1, 'img2ch': color_ui_2}

    def process_operation(self, image_dict, data_dict):
        input_img_1 = image_dict[self.input_image_name[0]]
        input_img_2 = image_dict[self.input_image_name[1]]
        output_img = np.zeros((input_img_1.shape[0],input_img_1.shape[1],3),'uint8')
        output_img[:,:,self.OperationParameter_dict['img1ch'].param_value] = np.maximum(input_img_1[:,:], output_img[:,:,self.OperationParameter_dict['img1ch'].param_value])
        output_img[:,:,self.OperationParameter_dict['img2ch'].param_value] = np.maximum(input_img_2[:,:], output_img[:,:,self.OperationParameter_dict['img2ch'].param_value])
        image_dict[self.output_image_name] = output_img

class AdaptiveThresholdOperation(ImageOperation):
    operation_category = "Filter"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_function = cv2.adaptiveThreshold
        athresh_methods_enum = (('Gaussian',cv2.ADAPTIVE_THRESH_GAUSSIAN_C),('Mean',cv2.ADAPTIVE_THRESH_MEAN_C))
        self.OperationParameter_dict = {'adaptiveMethod':OperationParameter('enum',1,0,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,"Type",param_enum=athresh_methods_enum),'blockSize': OperationParameter('int',1,3,55,5,'K','isOdd'),'C':OperationParameter('int',1,-255,255,0,"Constant to add")}
        # athresh_img = cv2.adaptiveThreshold(gauss_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,neighbor,thresh_c)
    
    def process_operation(self, image_dict, data_dict, **kwargs):
        return super().process_operation(image_dict, data_dict, maxValue=255, thresholdType=cv2.THRESH_BINARY, **kwargs)
    
class MorphologyOperation(ImageOperation):
    operation_category = "Filter"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_function = cv2.morphologyEx
        morph_enum = (('Open',cv2.MORPH_OPEN),('Close',cv2.MORPH_CLOSE))
        kernel_enum = (('Rectangle', cv2.MORPH_RECT),('Cross', cv2.MORPH_CROSS),('Ellipse', cv2.MORPH_ELLIPSE))
        self.OperationParameter_dict = {'k_size':OperationParameter('int',1,1,55,5,"Kernel Size",'isOdd'),'op':OperationParameter('enum',1,0,1,cv2.MORPH_OPEN,"Morph. Type",param_enum=morph_enum),'kernel': OperationParameter('enum',1,0,2,cv2.MORPH_RECT, "Kernel Shape", param_enum=kernel_enum)}

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        k_size = parameter_arguments.pop('k_size')
        kernel_shape = parameter_arguments.pop('kernel')
        morph_kernel = cv2.getStructuringElement(kernel_shape,(k_size,k_size))
        image_dict[self.output_image_name] = self.cv_function(image_dict[self.input_image_name], kernel=morph_kernel, **parameter_arguments, **kwargs)

class ContourOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict = {"thickness": OperationParameter('int',1,-1,200,3), "color": OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))}
        self.OperationParameter_dict['mode'] = OperationParameter('enum',1,0,4,cv2.RETR_LIST,"Retrieval Mode",param_enum=(("External",cv2.RETR_EXTERNAL),("List",cv2.RETR_LIST),("Components", cv2.RETR_CCOMP),("Tree",cv2.RETR_TREE)))
    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        contours, hierarchy = cv2.findContours(image_dict[self.input_image_name], parameter_arguments['mode'] ,cv2.CHAIN_APPROX_SIMPLE)
        image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
        cv2.drawContours(image_dict[self.output_image_name], contours, -1, parameter_arguments['color'],parameter_arguments['thickness'])
        data_dict[self.output_image_name]['contours'] = contours

class FillLargestContourOperation(ImageOperation):
    operation_category = "Draw"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict = {"thickness": OperationParameter('int',1,-1,200,-1), "color": OperationParameter('int',3,(0,0,0),(255,255,255),(255,255,255),('blue','green','red'))}

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        contours, hierarchy = cv2.findContours(image_dict[self.input_image_name], cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contourAreas = list(map(cv2.contourArea, contours))
        largest_contour = contours[np.array(contourAreas).argmax()]
        image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
        cv2.drawContours(image_dict[self.output_image_name], [largest_contour], -1, parameter_arguments['color'],parameter_arguments['thickness'])
        image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.output_image_name],cv2.COLOR_BGR2GRAY)
        data_dict[self.output_image_name]['largest_contour'] = largest_contour

class PolyContourOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 2 # Sets up input_image_name as list of length 2 once param controls exist, overwrites 'base_image' string 
        self.OperationParameter_dict = {"rel_epsilon": OperationParameter('float',1,0,1,0.005,"Rel. Epsilon"),"thickness": OperationParameter('int',1,0,10,3), "color": OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))}

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        contours = []
        # First image is drawn on, second image holds data
        for contour in data_dict[self.input_image_name[1]]['contours']:
            epsilon = self.OperationParameter_dict['rel_epsilon'].param_value*cv2.arcLength(contour,True)
            contours.append(cv2.approxPolyDP(contour,epsilon,True))

        if(len(image_dict[self.input_image_name[0]].shape)<3):
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name[0]],cv2.COLOR_GRAY2RGB)
        else:
            image_dict[self.output_image_name] = image_dict[self.input_image_name[0]].copy()

        cv2.drawContours(image_dict[self.output_image_name], contours, -1, parameter_arguments['color'],parameter_arguments['thickness'])
        data_dict[self.output_image_name]['contours'] = contours
    

class TransferContourOperation(ImageOperation):
    operation_category = "Draw"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 2 # Sets up input_image_name as list of length 2 once param controls exist, overwrites 'base_image' string 
        self.OperationParameter_dict = {"thickness": OperationParameter('int',1,-1,200,3), "color": OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))}

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        contours = data_dict[self.input_image_name[1]]['contours']

        if(len(image_dict[self.input_image_name[0]].shape) < 3): # Grayscale
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name[0]],cv2.COLOR_GRAY2RGB)
        else:
            image_dict[self.output_image_name] = image_dict[self.input_image_name[0]].copy()

        cv2.drawContours(image_dict[self.output_image_name], contours, -1, parameter_arguments['color'],parameter_arguments['thickness'])
        data_dict[self.output_image_name]['contours'] = contours

# class CannyEdgeOperation(ImageOperation):
#     pass

class CornerOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict['maxCorners'] = OperationParameter('int',1,0,200,25,"Max Corners")
        self.OperationParameter_dict['qualityLevel'] = OperationParameter('float',1,0,1,0.05,"Quality level")
        self.OperationParameter_dict['minDistance'] = OperationParameter('int',1,1,500,10,"Min Distance")
        self.OperationParameter_dict['blockSize'] = OperationParameter('int',1,1,300,15,"Block Size")
        self.OperationParameter_dict['useHarrisDetector'] = OperationParameter('enum',1,0,1,1,"Detector",param_enum=(("Harris Detector",True),("Min. Eigenvalue",False)))
        self.OperationParameter_dict['k'] = OperationParameter('float',1,0,1,0.04,"Harris Parameter")
        self.OperationParameter_dict["thickness"] = OperationParameter('int',1,0,100,3)
        self.OperationParameter_dict["color"] = OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        pt_size = parameter_arguments.pop('thickness')
        color = parameter_arguments.pop('color')
        points = cv2.goodFeaturesToTrack(image_dict[self.input_image_name],**parameter_arguments)
        if points is not None:
            points = np.squeeze(points.astype(int))

        if(len(image_dict[self.input_image_name].shape) < 3): # Grayscale
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
        else:
            image_dict[self.output_image_name] = image_dict[self.input_image_name].copy()
        for pt in points:
            x,y = pt.ravel()
            cv2.circle(image_dict[self.output_image_name],(x,y),pt_size,color, -1)

        data_dict[self.output_image_name]['points'] = points
    
class CornerMaskedOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict['maxCorners'] = OperationParameter('int',1,0,200,25,"Max Corners")
        self.OperationParameter_dict['qualityLevel'] = OperationParameter('float',1,0,1,0.05,"Quality level")
        self.OperationParameter_dict['minDistance'] = OperationParameter('int',1,1,500,10,"Min Distance")
        self.OperationParameter_dict['blockSize'] = OperationParameter('int',1,1,300,15,"Block Size")
        self.OperationParameter_dict['useHarrisDetector'] = OperationParameter('enum',1,0,1,1,"Detector",param_enum=(("Harris Detector",True),("Min. Eigenvalue",False)))
        self.OperationParameter_dict['k'] = OperationParameter('float',1,0,1,0.04,"Harris Parameter")
        self.OperationParameter_dict["thickness"] = OperationParameter('int',1,0,100,3)
        self.OperationParameter_dict["color"] = OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))
        self.input_image_count = 2
    
    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        parameter_arguments['mask'] = image_dict[self.input_image_name[1]]
        pt_size = parameter_arguments.pop('thickness')
        color = parameter_arguments.pop('color')
        points = cv2.goodFeaturesToTrack(image_dict[self.input_image_name[0]],**parameter_arguments)
        if points is not None:
            points = np.squeeze(points.astype(int))

        if(len(image_dict[self.input_image_name[0]].shape) < 3): # Grayscale
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name[0]],cv2.COLOR_GRAY2RGB)
        else:
            image_dict[self.output_image_name] = image_dict[self.input_image_name[0]].copy()
        for pt in points:
            x,y = pt.ravel()
            cv2.circle(image_dict[self.output_image_name],(x,y),pt_size,color, -1)

        data_dict[self.output_image_name]['points'] = points
    
class EuclidTransformOperation(ImageOperation):
    operation_category = "Transform"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict['rotation'] = OperationParameter('float',1,-360,360,0,"Degrees")
        self.OperationParameter_dict['translation'] = OperationParameter('int',2,(-1000,-1000),(1000,1000),(0,0),("X","Y"))
        # X is inverted from expected (positive values go left instead of right), may want to adjust in controls

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        # transformation = skimage.transform.EuclideanTransform(rotation=np.radians(parameter_arguments['rotation']),translation=parameter_arguments['translation'])
        # image_dict[self.output_image_name] = skimage.util.img_as_ubyte(skimage.transform.warp(image_dict[self.input_image_name],transformation))
        transformation = skimage.transform.EuclideanTransform(rotation=np.radians(parameter_arguments['rotation']),translation=parameter_arguments['translation']).params[:2,:]
        dst_size = (image_dict[self.input_image_name].shape[1], image_dict[self.input_image_name].shape[0])
        image_dict[self.output_image_name] = cv2.warpAffine(image_dict[self.input_image_name],transformation,dst_size,flags=cv2.WARP_INVERSE_MAP)

    def update_param_limits(self, image_dict):
        if not self.input_image_exists(image_dict):
            return
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        self.set_param_limits('translation', param_max=img_size, param_min=[-length for length in img_size])

class MatchResolutionOperation(ImageOperation):
    operation_category = "Resize"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 2

    def process_operation(self, image_dict, data_dict, **kwargs):
        padding = np.subtract(image_dict[self.input_image_name[1]].shape[:2], image_dict[self.input_image_name[0]].shape[:2])
        if np.min(padding) < 0:
            return
        image_dict[self.output_image_name] = cv2.copyMakeBorder(image_dict[self.input_image_name[0]],0,padding[0],0,padding[1],borderType=cv2.BORDER_CONSTANT,value=0)

class BlankCopyResolutionOperation(ImageOperation):
    operation_category = "Generate"
    # Creates a blank image based off resolution of input image
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict['channels'] = OperationParameter('enum',1,1,3,1,"Color",param_enum=(("3 Channel Color",3),("Grayscale",1)))
    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        if parameter_arguments['channels'] == 1:
            output_img = np.zeros((image_dict[self.input_image_name].shape[0],image_dict[self.input_image_name].shape[1]),'uint8')
        else:
            output_img = np.zeros((image_dict[self.input_image_name].shape[0],image_dict[self.input_image_name].shape[1],3),'uint8')
        image_dict[self.output_image_name] = output_img

class BlankResolutionOperation(ImageOperation):
    operation_category = "Generate"
    # Creates a blank image based off resolution of input image
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict['channels'] = OperationParameter('enum',1,1,3,1,"Color",param_enum=(("3 Channel Color",3),("Grayscale",1)))
        self.OperationParameter_dict['resolution'] = OperationParameter('int',2,(0,0),(50000,50000),(100,100),("Width","Height"))
    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        if parameter_arguments['channels'] == 1:
            output_img = np.zeros((parameter_arguments['resolution'][1],parameter_arguments['resolution'][0]),'uint8')
        else:
            output_img = np.zeros((parameter_arguments['resolution'][1],parameter_arguments['resolution'][0],3),'uint8')
        image_dict[self.output_image_name] = output_img

class CountOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_operation(self, image_dict, data_dict, **kwargs):
        data_dict[self.output_image_name]['pixel_area'] = cv2.countNonZero(image_dict[self.input_image_name])
        print(data_dict[self.output_image_name]['pixel_area'])
        image_dict[self.output_image_name] = image_dict[self.input_image_name].copy()

class CountMaskOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        maskFrameTL = OperationParameter('point',2,(0,0),(100,100),(0,0),('left','top'))
        maskFrameBR = OperationParameter('point',2,(0,0),(100,100),(0,0),('right','bottom'))
        self.OperationParameter_dict['pt1'] = maskFrameTL
        self.OperationParameter_dict['pt2'] = maskFrameBR

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        topLeft = parameter_arguments['pt1']
        bottomRight = parameter_arguments['pt2']
        img = image_dict[self.input_image_name]

        data_dict[self.output_image_name]['pixel_area'] = cv2.countNonZero(img[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]])

        mask = np.zeros((image_dict[self.input_image_name].shape[0],img.shape[1]),'uint8')
        cv2.rectangle(mask,topLeft,bottomRight,(255,255,255),-1)
        img_channels = img.shape[2] if len(img.shape) > 2 else 1
        if img_channels == 3:
            mask = np.dstack((mask, mask, mask))
        image_dict[self.output_image_name] = cv2.bitwise_and(img,mask)
        
    def update_param_limits(self, image_dict):
        if not self.input_image_exists(image_dict):
            return
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        self.set_param_limits('pt1', param_max=img_size)
        self.set_param_limits('pt2', param_max=img_size)

class CountColorOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict['channel'] = OperationParameter('enum',1,0,2,0,"Color",param_enum=(("Blue",0),("Green",1),("Red",2)))

    def process_operation(self, image_dict, data_dict, **kwargs):
        #Isolate
        parameter_arguments = self.make_param_dict()
        data_dict[self.output_image_name]['pixel_area'] = cv2.countNonZero(image_dict[self.input_image_name][:,:,parameter_arguments['channel']])
        print(data_dict[self.output_image_name]['pixel_area'])
        image_dict[self.output_image_name] = image_dict[self.input_image_name].copy()
        
class MeasureWidthOperation(ImageOperation):
    operation_category = "Data"
    # Measures width of 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict = {"thickness": OperationParameter('int',1,-1,200,3), "color": OperationParameter('int',3,(0,0,0),(255,255,255),(255,0,0),('blue','green','red'))}
        self.OperationParameter_dict["scale"] = OperationParameter('float',1,0.0,500,1,"px/mm",float_precision=3)
    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        contours, hierarchy = cv2.findContours(image_dict[self.input_image_name], cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if len(contours) < 1:
            data_dict[self.output_image_name]['min_distance'] = 0
            data_dict[self.output_image_name]['fit_distance'] = 0
            data_dict[self.output_image_name]['max_distance'] = 0
            data_dict[self.output_image_name]['left_fit_roughness'] = 0
            data_dict[self.output_image_name]['right_fit_roughness'] = 0
            data_dict[self.output_image_name]['left_fit_stddev'] = 0
            data_dict[self.output_image_name]['right_fit_stddev'] = 0
            data_dict[self.output_image_name]['fit_roughness'] = 0
            data_dict[self.output_image_name]['fit_stddev'] = 0
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
            return
        
        # Use largest contour's centroid to determine the dividing point between left and right
        contourAreas = list(map(cv2.contourArea, contours))
        largest_contour = np.squeeze(contours[np.array(contourAreas).argmax()])
        cnt_moments = cv2.moments(largest_contour)
        horizontal_centre = int(cnt_moments['m10']/cnt_moments['m00'])

        # Use all contours points from all contours
        all_cnt_pts = np.concatenate([np.squeeze(cnt) for cnt in contours],axis=0)

        img_shape = image_dict[self.input_image_name].shape
        height, width, *_ = img_shape
        margin = 5

        left_points = np.array([pt for pt in all_cnt_pts if pt[0] < horizontal_centre and pt[1] > margin and pt[1] < height-margin])
        right_points = np.array([pt for pt in all_cnt_pts if pt[0] > horizontal_centre and pt[1] > margin and pt[1] < height-margin])
        
        if left_points.size == 0 or right_points.size == 0:
            data_dict[self.output_image_name]['distance'] = 0
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
            return

        left_min = np.max(left_points[:,0])
        left_max = np.min(left_points[:,0])
        left_error = left_points[:,0] - np.mean(left_points[:,0])
        left_stddev = np.std(left_error)
        left_roughness = np.sum(np.abs(left_error),0) / len(left_error)

        right_min = np.min(right_points[:,0])
        right_max = np.max(right_points[:,0])
        right_error = right_points[:,0] - np.mean(right_points[:,0])
        right_stddev = np.std(right_error)
        right_roughness = np.sum(np.abs(right_error),0) / len(right_error)

        combined_error = np.concatenate((left_error, right_error))
        fit_roughness = np.sum(np.abs(combined_error),0) / len(combined_error)
        fit_stddev = np.std(combined_error)

        left_fit = np.mean(left_points[:,0])
        right_fit = np.mean(right_points[:,0])

        data_dict[self.output_image_name]['min_distance'] = (right_min - left_min) / parameter_arguments['scale']
        data_dict[self.output_image_name]['fit_distance'] = (right_fit - left_fit) / parameter_arguments['scale']
        data_dict[self.output_image_name]['max_distance'] = (right_max - left_max) / parameter_arguments['scale']

        data_dict[self.output_image_name]['left_fit_roughness'] = left_roughness / parameter_arguments['scale']
        data_dict[self.output_image_name]['right_fit_roughness'] = right_roughness / parameter_arguments['scale']
        data_dict[self.output_image_name]['left_fit_stddev'] = left_stddev / parameter_arguments['scale']
        data_dict[self.output_image_name]['right_fit_stddev'] = right_stddev / parameter_arguments['scale']

        data_dict[self.output_image_name]['fit_roughness'] = fit_roughness / parameter_arguments['scale']
        data_dict[self.output_image_name]['fit_stddev'] = fit_stddev / parameter_arguments['scale']

        # print(f'{left_pos}, {right_pos}, {right_pos-left_pos}')

        image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
        cv2.line(image_dict[self.output_image_name], (int(left_min),0), (int(left_min),height), (255,0,0), parameter_arguments['thickness'])
        cv2.line(image_dict[self.output_image_name], (int(right_min),0), (int(right_min),height), (255,0,0), parameter_arguments['thickness'])
        cv2.line(image_dict[self.output_image_name], (int(left_max),0), (int(left_max),height), (0,0,255), parameter_arguments['thickness'])
        cv2.line(image_dict[self.output_image_name], (int(right_max),0), (int(right_max),height), (0,0,255), parameter_arguments['thickness'])
        cv2.line(image_dict[self.output_image_name], (int(left_fit),0), (int(left_fit),height), (0,255,0), parameter_arguments['thickness'])
        cv2.line(image_dict[self.output_image_name], (int(right_fit),0), (int(right_fit),height), (0,255,0), parameter_arguments['thickness'])
        # cv2.line(image_dict[self.output_image_name], (p1x,p1y), (p2x,p2y), parameter_arguments['color'],parameter_arguments['thickness'])
        image_dict[self.output_image_name] = image_dict[self.output_image_name]

class MeasureCircleOperation(ImageOperation):
    operation_category = "Data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OperationParameter_dict = {"thickness": OperationParameter('int',1,-1,200,3)}
        self.OperationParameter_dict["scale"] = OperationParameter('float',1,0.0,500,1,"px/mm",float_precision=3)

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        contours, hierarchy = cv2.findContours(image_dict[self.input_image_name], cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if len(contours) < 1:
            data_dict[self.output_image_name]['min_diameter'] = 0
            data_dict[self.output_image_name]['fit_diameter'] = 0
            data_dict[self.output_image_name]['max_diameter'] = 0
            data_dict[self.output_image_name]['fit_roughness'] = 0
            data_dict[self.output_image_name]['fit_stddev'] = 0
            image_dict[self.output_image_name] = cv2.cvtColor(image_dict[self.input_image_name],cv2.COLOR_GRAY2RGB)
            return
        
        contourAreas = list(map(cv2.contourArea, contours))
        largest_contour = np.squeeze(contours[np.array(contourAreas).argmax()])
        img = image_dict[self.input_image_name]
        img_shape = img.shape
        height, width, *_ = img_shape
        
        class CoopeCircle():
            def estimate(self, data):
                self.B = np.column_stack((data,np.ones(data.shape[0],int))) #Augment data array
                self.d = data[:,0]**2 + data[:,1]**2
                self.y, residuals, rank, singular = np.linalg.lstsq(self.B, self.d, rcond=None) # Run linear least squares optimization
                self.centre = self.y[:2] / 2
                self.radius = (self.y[2]+self.centre.T@self.centre)**(1/2)
                return True

            def residuals(self, data):
                return (self.radius - np.sum(((self.centre - data) ** 2), 1)**(1/2))**2 # Distance squared
            
            def error(self, data):
                return self.radius - np.sum(((self.centre - data) ** 2), 1)**(1/2)
            
            def stdDev(self, data):
                return np.std(self.error(data)) # Standard deviation of error distance
            
            def contourRoughness(self, data):
                return np.sum(np.abs(self.error(data)),0) / len(data) # Average error of the contour (total sum of error / contour length) in pixels 
            
        # Inscribed Circle
        dist_trans = cv2.distanceTransform(img, cv2.DIST_L2,cv2.DIST_MASK_5)
        _, inner_radius, _, inner_centre  = cv2.minMaxLoc(dist_trans)

        # Circumscribed Circle
        (outer_centre, outer_radius) = cv2.minEnclosingCircle(largest_contour)

        # Fit Circle
        # Use RANSAC and Coope Linear Least Squares Circle fit
        # model_ransac, inliers = skimage.measure.ransac(largest_contour, CoopeCircle, 5, (cv2.contourArea(largest_contour)/np.pi)**(1/2)*0.1, max_trials=5000)
        # fit_centre = model_ransac.centre
        # fit_radius = model_ransac.radius

        # Use only Linear Least Squares Fit
        model = CoopeCircle()
        model.estimate(largest_contour)
        fit_centre = model.centre
        fit_radius = model.radius
        fit_roughness = model.contourRoughness(largest_contour)
        fit_stddev = model.stdDev(largest_contour)
        # Output data
        data_dict[self.output_image_name]['min_diameter'] = inner_radius*2 / parameter_arguments['scale']
        data_dict[self.output_image_name]['fit_diameter'] = fit_radius*2 / parameter_arguments['scale']
        data_dict[self.output_image_name]['max_diameter'] = outer_radius*2 / parameter_arguments['scale']
        data_dict[self.output_image_name]['fit_roughness'] = fit_roughness / parameter_arguments['scale']
        data_dict[self.output_image_name]['fit_stddev'] = fit_stddev / parameter_arguments['scale']
        
        # Convert to int for drawing
        inner_centre = np.array(inner_centre).astype(int)
        inner_radius = int(inner_radius)
        outer_centre = np.array(outer_centre).astype(int)
        outer_radius = int(outer_radius)
        fit_centre = fit_centre.astype(int)
        fit_radius = int(fit_radius)
        # Output image 
        circle_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(circle_img, outer_centre, outer_radius, (0,0,255),parameter_arguments['thickness'])
        cv2.circle(circle_img, fit_centre, fit_radius, (0,255,0),parameter_arguments['thickness'])
        cv2.circle(circle_img, inner_centre, inner_radius, (255,0,0),parameter_arguments['thickness'])
        image_dict[self.output_image_name] = circle_img

class MaskOperation(ImageOperation):
    operation_category = "Generate"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        maskFrameTL = OperationParameter('point',2,(0,0),(100,100),(0,0),('left','top'))
        maskFrameBR = OperationParameter('point',2,(0,0),(100,100),(0,0),('right','bottom'))
        self.OperationParameter_dict['pt1'] = maskFrameTL
        self.OperationParameter_dict['pt2'] = maskFrameBR

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        topLeft = parameter_arguments['pt1']
        bottomRight = parameter_arguments['pt2']
        mask = np.zeros((image_dict[self.input_image_name].shape[0],image_dict[self.input_image_name].shape[1]),'uint8')
        cv2.rectangle(mask,topLeft,bottomRight,(255,255,255),-1)
        img = image_dict[self.input_image_name]
        img_channels = img.shape[2] if len(img.shape) > 2 else 1
        if img_channels == 3:
            mask = np.dstack((mask, mask, mask))
        image_dict[self.output_image_name] = cv2.bitwise_and(image_dict[self.input_image_name],mask)

    def update_param_limits(self, image_dict):
        if not self.input_image_exists(image_dict):
            return
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        self.set_param_limits('pt1', param_max=img_size)
        self.set_param_limits('pt2', param_max=img_size)

class FrameMaskOperation(ImageOperation):
    operation_category = "Generate"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        maskFrameTL = OperationParameter('point',2,(0,0),(100,100),(0,0),('left','top'))
        maskFrameBR = OperationParameter('point',2,(0,0),(100,100),(0,0),('right','bottom'))
        self.OperationParameter_dict['topLeftPt'] = maskFrameTL
        self.OperationParameter_dict['bottomRightPt'] = maskFrameBR

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        image_dict[self.output_image_name] = np.zeros((image_dict[self.input_image_name].shape[0],image_dict[self.input_image_name].shape[1]),'uint8')
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        topLeft = parameter_arguments['topLeftPt']
        bottomRight = parameter_arguments['bottomRightPt']
        topRight = (bottomRight[0],topLeft[1])
        bottomLeft = (topLeft[0],bottomRight[1])
        cv2.rectangle(image_dict[self.output_image_name],topLeft,(0,0),(255,255,255),-1)
        cv2.rectangle(image_dict[self.output_image_name],bottomLeft,(0,img_size[1]),(255,255,255),-1)
        cv2.rectangle(image_dict[self.output_image_name],topRight,(img_size[0],0),(255,255,255),-1)
        cv2.rectangle(image_dict[self.output_image_name],bottomRight,(img_size[0],img_size[1]),(255,255,255),-1)

    def update_param_limits(self, image_dict):
        if not self.input_image_exists(image_dict):
            return
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        self.set_param_limits('topLeftPt', param_max=img_size)
        self.set_param_limits('bottomRightPt', param_max=img_size)

class CropOperation(ImageOperation):
    operation_category = "Resize"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        maskFrameTL = OperationParameter('point',2,(0,0),(100,100),(0,0),('left','top'))
        maskFrameBR = OperationParameter('point',2,(0,0),(100,100),(0,0),('right','bottom'))
        self.OperationParameter_dict['pt1'] = maskFrameTL
        self.OperationParameter_dict['pt2'] = maskFrameBR

    def process_operation(self, image_dict, data_dict, **kwargs):
        parameter_arguments = self.make_param_dict()
        topLeft = parameter_arguments['pt1']
        bottomRight = parameter_arguments['pt2']
        image_dict[self.output_image_name] = image_dict[self.input_image_name][topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
        
    def update_param_limits(self, image_dict):
        if not self.input_image_exists(image_dict):
            return
        img_size = np.flip(image_dict[self.input_image_name].shape[:2])
        self.set_param_limits('pt1', param_max=img_size)
        self.set_param_limits('pt2', param_max=img_size)

class RegistrationOperation(ImageOperation):
    operation_category = "Transform"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_image_count = 3
        self.OperationParameter_dict['manual_rotation'] = OperationParameter('float',1,-180,180,0,"Degrees")
        self.OperationParameter_dict['manual_translation'] = OperationParameter('int',2,(-1000,-1000),(1000,1000),(0,0),("X","Y"))

    def process_operation(self, image_dict, data_dict, **kwargs):
        def sortPoints(points: np.ndarray):
            pt_sum = np.sum(points,1)
            pt_diff = np.diff(points,1)
            topLeft = pt_sum.argmin()
            bottomRight = pt_sum.argmax()

            topRight = pt_diff.argmin()
            bottomLeft = pt_diff.argmax()
            # Sorting fails if points have same sum or same difference, okay for this application
            
            return points[(topLeft, bottomLeft, bottomRight, topRight), :]
        
        # 4 image inputs (image1, image1data (points), image2, image2data (points))
        parameter_arguments = self.make_param_dict()
        img_src = image_dict[self.input_image_name[0]]
        pts_src = sortPoints(data_dict[self.input_image_name[1]]['points'])
        pts_dst = sortPoints(data_dict[self.input_image_name[2]]['points'])
        # Need to arrange points, should only have 4 points, one for each corner of part
        transform = skimage.transform.EuclideanTransform()
        # Keep only 3 relative points, prevents 180 degree rotation, locks relative rotation of cad
        # transform.estimate(pts_dst[:3,:], pts_src[:3,:])
        transform.estimate(pts_dst, pts_src)
        transform_manual = skimage.transform.EuclideanTransform(rotation=np.radians(parameter_arguments['manual_rotation']),translation=parameter_arguments['manual_translation'])
        transform_total = skimage.transform.EuclideanTransform(matrix=transform_manual.params @ transform.params)
        data_dict[self.output_image_name]['transform'] = {'rotation': transform_total.rotation, 'translation': transform_total.translation}
        image_dict[self.output_image_name] = skimage.util.img_as_ubyte(skimage.transform.warp(img_src, transform_total))