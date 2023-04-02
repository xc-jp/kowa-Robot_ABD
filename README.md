# kowa-Robot_ABD
* Purpose of work: Abnormality Detection using 2D picking model
* Detailed description:
 Implementing a python script (test_abnormality_detection.py) that receives as input arguments:
    - the trained 2D picking model path
    - the objects' image that is subject for evaluation (normally a camera shot showing objects in the working scene)
    - the allowed regions' image, which is used as reference map for identifying the prohibited and allowed zones for the objects; this map is a B&W image where white pixels indicate the allowed regions.
* Destined output:
    - returning (and printing) the grasping_inference results as output: a list of dictionaries, where each dictionary refers to a detected object; main object details: coordinates xy, angle, and judgement of the position: inside/outside allowed region (in boolean value)
    - visualizing the detected objects on the allowed_regions map, which is saving as an Image file.

* Necessary Presets:
    * Data acquisition: model and sample image files are available on XC's NAS server
        - Grasping Model: \\192.168.100.95\pub\A4_Kowa-Optronics\6_2D_picking\mlserver_test\kowa_infer.zip
        - Test Images: \\192.168.100.95\pub\A4_Kowa-Optronics\6_2D_picking\mlserver_test\box_small_amount\Test\images
    * PyTorchGrasping submodule:
        - add https://github.com/xc-jp/PyTorchGrasping/tree/kowa/v1.2.1 as submodule
        - install https://github.com/xc-jp/PyTorchGrasping/tree/kowa/v1.2.1 requirements
        - set the Python Path for including the submodule on the interpreter:
            - in Workspace Settings (JSON): 
                {
                    "python.analysis.extraPaths": ["./,   "./PyTorchGrasping"]
                }
            - on Terminal: set PYTHONPATH=.;.\PyTorchGrasping;

* WORK STILL UNDER CONSTRUCTION:
    - evaluating whether or not objects are in allowed regions
    - outputting a message if an object is outside allowed region
