# kowa-Robot_ABD
Implementing a python script (test_abnormality_detection.py) that receives the model and image paths,  and returns the grasping_inference results:
a list of center coordinates of the the detected objects, and their visualization saved as an image file.
Consequently, the script detects the objects that are outside the allowed regions (depending on to the input allowed_regions map), and prints them as an output.

Data acquisition: model and image files on NAS server
    - Grasping Model: \\192.168.100.95\pub\A4_Kowa-Optronics\6_2D_picking\mlserver_test\kowa_infer.zip
    - Test Images: \\192.168.100.95\pub\A4_Kowa-Optronics\6_2D_picking\mlserver_test\box_small_amount\Test\images
    - allowed_regions: a B&W image (white for allowed) having the same dimensions an Test Images