# kowa-Robot_ABD
* Purpose of work: Abnormality Detection using 2D picking model
* Detailed description:
 Implementing a python script (test_abnormality_detection.py) that receives as input arguments:
    - the trained 2D picking model path
    - the objects' image that is subject for evaluation (normally a camera shot showing objects in the working scene)
    - the allowed regions' image, which is used as reference map for identifying the prohibited and allowed zones for the objects; this map is a B&W image where white pixels indicate the allowed regions on the scene
* Destined output:
    - returning (and printing) the grasping_inference results as output: a list of dictionaries, where each dictionary refers to a detected object; main object details: coordinates xy, angle, and judgement of the position: inside/outside allowed region (in boolean value)
    - visualizing the detected objects on the allowed_regions map, and saving this visualization as an Image file.

* Necessary Presets:
    * Data acquisition: model and sample image files are available on XC's NAS server:
        - Grasping Model: \\192.168.100.95\pub\A4_Kowa-Optronics\6_2D_picking\mlserver_test\kowa_infer.zip
        - _PS_: The user needs to unzip the file, and the model path should point to the "grasping" folder
    * PyTorchGrasping submodule:
        - initialize the submodule after cloning the project: `git submodule update --init --recursive`
        - install https://github.com/xc-jp/PyTorchGrasping/tree/kowa/v1.2.1 requirements
        - set PythonPath for including the submodule on the interpreter:
            - on Terminal: `set PYTHONPATH=.;.\PyTorchGrasping;`
            - on Windows DOS: `set PYTHONPATH=.;.\PyTorchGrasping;`
            - on Powershell: `$env:PYTHONPATH+='.;.\PyTorchGrasping'`
            - on Ubuntu or Mac terminal: `export PYTHONPATH=.:./PyTorchGrasping:`
        - run the script as the following examples:
          - No GPU: `python .\test_abnormality_detection.py <path\to\>\kowa_infer\grasping .\samples\test_images\00000001_clear_center.jpg .\samples\allowed_regions.png`
          - With GPU (you must have cuda installed on your environment): `python .\test_abnormality_detection.py <path\to\>\kowa_infer\grasping .\samples\test_images\00000001_clear_center.jpg .\samples\allowed_regions.png --gpu`


* WORK STILL UNDER CONSTRUCTION:
    - evaluating whether or not objects are in allowed regions
    - outputting a message if an object is outside allowed region
