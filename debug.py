import os
import sys

from PIL import Image

 
class DebuggerParams:
    def __init__(self, name, ext):
        self.count = 0
        self.name = name
        self.ext = ext

    def next(self):
        self.count += 1

    def get_file_save_path(self, func_name, custom_name=""):
        self.next()
        if custom_name != "":
            return os.path.join(self.name, f"{self.count}_{self.name}_{custom_name}{self.ext}")
        return os.path.join(self.name, f"{self.count}_{self.name}{self.ext}")
    
    
class Debugger:
    def __init__(self, is_enabled, debug_path):
        self.is_enabled = is_enabled
        self.debug_path = debug_path

        # create root folder if not exists
        if is_enabled is True:
            os.makedirs(self.debug_path, exist_ok=True)

    def save_image(self, debug_params, image):
        if debug_params is None or image is None:
            return

        if self.is_enabled is True:
            traceback_func = sys._getframe(2).f_code.co_name
            if traceback_func == "save_images":
                traceback_func = sys._getframe(3).f_code.co_name

            debug_out_path = os.path.join(
                    self.debug_path, debug_params.get_file_save_path(traceback_func)
                )

            if debug_params.count == 1:
                save_folder_path = os.path.join(self.debug_path, debug_params.name)
                os.makedirs(save_folder_path, exist_ok=True)

            image.save(debug_out_path)

    def save_images(self, debug_params, images):
        for img in images:
            self.save_image(debug_params, img)


