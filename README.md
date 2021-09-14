# passport-photo-generator
Generate photos that satisfy US Passport requriements using Node.js, Express.js, and HTML to display the website with python, opencv, and U-2 Net for image processing.

Directions for use:

1) Download the full Salient Object Detection U-2 Net model from https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view and add to your main directory 

OR 

2) Modify model_name in rotatedetect.py to "u2netp" to use the smaller already included model(u2netp.pth) instead.

## U-2 Net Citation:

title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection}

author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin}

journal = {Pattern Recognition}

volume = {106}

pages = {107404},

year = {2020}
