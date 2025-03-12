from vedo import *
import math

import datetime
from vedo.applications import IsosurfaceBrowser
from generate_deepdrr import Generate 

import pandas as pd
import sys


# initiate deepdrr initiation
g = Generate(sys.argv[1])
g.empty_file()

# initiate parameters
x = datetime.datetime.now()

ct_name = sys.argv[1]
operator = sys.argv[2]
case_name = sys.argv[1].split('/')[1].split('_')[0]

target = 0
order = 1

# Create a timer thread
timer_thread = None
time_pre = 0

# position starts
position_list = ["skull", 
                 "right_humeral_head", 
                 "left_humeral_head", 
                 "right_scapula", 
                 "left_scapula",
                 "right_elbow",
                 "left_elbow",
                 "right_wrist",
                 "left_wrist", 
                 "T1", 
                 "carina", 
                 "right_hemidiaphragm",
                 "left_hemidiaphragm",
                 "T12", 
                 "L5", 
                 "right_iliac_crest", 
                 "left_iliac_crest",
                 "pubic_symphysis",
                 "right_femoral_head",
                 "left_femoral_head"]


coordinate_to_send = []

def buttonfunc_na():
    """
    buttonfunc_na

    descriptions
    ---------------------------------
    Function for na button: it will send the post request and switch to the next task
    """        

    global order
    global coordinate_to_send
    if order < 21:
        order+=1
        coordinate_to_send.append(None)

        if order != 21:
            plt.at(2).show(Picture(f"result_text/{position_list[order - 1]}.png"),axes=0,zoom=1.9)


def buttonfunc_g():
    """
    buttonfunc_g

    descriptions
    ---------------------------------
    Function for g button
    """    
    global cam_distance

    loc = circle.GetPosition()


    sin_rad_alpha = 0
    sin_rad_beta = 0
    
    g.deepdrr_run(loc[2] - center[2], loc[0] - center[0],loc[1] - center[1], math.asin(sin_rad_alpha),math.asin(sin_rad_beta))
    plt.at(3).show(Picture("projector.png"),axes=0, zoom=1.5)

def buttonfunc_back():
    """
    buttonfunc_back

    descriptions
    ---------------------------------
    Function for back button: it will send the post request and switch to the back task
    """        
    global order
    global coordinate_to_send
    if order != 1:

        # x, y, z, a, b, position (1-11), ct_name (file name), operator_id, case name
        coordinate_to_send = coordinate_to_send[:-1]

        order -=1
        plt.at(2).show(Picture(f"result_text/{position_list[order - 1]}.png"),axes=0,zoom=1.9)
def buttonfunc_next():
    """
    buttonfunc_next

    descriptions
    ---------------------------------
    Function for next button: it will send the post request and switch to the next task
    """        
    global order
    global coordinate_to_send
    if order < 21:

        global cam_distance

        loc = circle.GetPosition()

        sin_rad_alpha = 0
        sin_rad_beta = 0

        # x, y, z, a, b, position (1-11), ct_name (file name), operator_id, case name
        coordinate_to_send.append((loc[2] - center[2], loc[0] - center[0], loc[1] - center[1], math.asin(sin_rad_alpha), math.asin(sin_rad_beta), order, ct_name, operator, case_name))

        order+=1
        if order != 21:
            plt.at(2).show(Picture(f"result_text/{position_list[order - 1]}.png"),axes=0,zoom=1.9)

def buttonfunc_finish():
    """
    buttonfunc_finish

    descriptions
    ---------------------------------
    Function for finish button: it will send the post request and switch to the next task
    """       
    global order 
    global coordinate_to_send
  
    if order == 21:

        df = pd.DataFrame(coordinate_to_send, columns=['x', 'y', 'z', 'a', 'b', 'landmark', 'file_name', 'operator', 'case_name'])
        output_file = "annotations.csv"

        if not os.path.exists(output_file):
            df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)
        exit()

def func(evt):
    msh = evt.actor
    event_at = evt.at
    if msh and event_at == 1:
    
        pos1 = plt.at(1).camera.GetPosition()
        foc1 = plt.at(1).camera.GetFocalPoint()

        cood = evt.picked3d

        circle.x(cood[0])
        circle.z(cood[2])
        box.x(cood[0])
        box.z(cood[2])

        plt.at(1).camera.SetFocalPoint([cood[0], foc1[1], cood[2]])
        plt.at(1).camera.SetPosition([cood[0], pos1[1], cood[2]])
        txt.text(f"({cood[0]:.2f}, {cood[2]:.2f})")

        plt.render()

        buttonfunc_g()
    if msh and event_at == 3:
        cood = evt.picked3d

        pos1 = plt.at(1).camera.GetPosition()
        foc1 = plt.at(1).camera.GetFocalPoint()

        circle_cood = circle.GetCenter()

        x_d = (cood[0] - (1536 / 2)) / (1536 * 3) * (y1 - y0)
        y_d = (cood[1] - (1536 / 2)) / (1536 * 3) * (y1 - y0)

        circle.x(circle_cood[0] + x_d)
        circle.z(circle_cood[2] - y_d)

        box.x(circle_cood[0] + x_d)
        box.z(circle_cood[2] - y_d)

        plt.at(1).camera.SetFocalPoint([circle_cood[0] + x_d, foc1[1], circle_cood[2] - y_d])
        plt.at(1).camera.SetPosition([circle_cood[0] + x_d, pos1[1], circle_cood[2] - y_d])
        txt.text(f"({circle_cood[0] + x_d:.2f}, {circle_cood[2] - y_d:.2f})")
        plt.render()

        buttonfunc_g()

# disable draging via mouse
settings.renderer_frame_width = 1
settings.enable_default_keyboard_callbacks = False

ct = Volume(ct_name, mapper="gpu")

ct.cmap("rainbow").alpha([0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.8, 1])
vol_arr = ct.tonumpy()

vol_arrc = np.zeros_like(vol_arr, dtype=np.uint8)
vol_arrc[(vol_arr > 123) & (vol_arr < 3000)] = 1
ct.mask(vol_arrc)
# substitute scalarbar3d to a 2d scalarbar
# get the demension/shape of the ct scans
x0, x1, y0, y1, z0, z1 = ct.bounds()

center = [(x1 - x0) / 2, (y1 - y0) / 2 , (z1 - z0) / 2]
cam_high = [x1 / 2,(y1 - y0)/1.1, z1 * (1 / 4)]
cam_side = [-(x1 - x0), (y1 - y0) * 2, (z1 - z0) / 2]

click = False

circle = Cylinder(pos = (cam_high[0], cam_high[1] / 1.39, cam_high[2]),
          r = 7,
          height = 20,
          alpha = 1,
           axis = (0, 1, 0), c='black')
box = Box(pos = (cam_high[0], cam_high[1], cam_high[2]),
          length=150,
          width=10,
          height=150,
          alpha = 0.95, c='red')
shape = [
    dict(bottomleft=(0,0), topright=(1,1), bg='grey'), # the full empty window
    dict(bottomleft=(0.01,0.07), topright=(0.6,0.99), bg='w'), # ct with box
    dict(bottomleft=(0.62,0.70), topright=(0.99,0.99), bg='w'), # instructions
    dict(bottomleft=(0.62,0.07), topright=(0.99,0.69), bg='w'), # x-ray view
]

# setup plotter
plt = Plotter(shape=shape, sharecam=False, size=(2400, 1500),title="carm simulator",bg='black')

txt = Text2D(f"({cam_high[0]:.2f}, {cam_high[2]:.2f})", pos="bottom-right",s=2, font='Brachium', c='black', bg='white',alpha=1)
plt.at(0).show(txt)
plt.at(1).show(Assembly([box, ct,circle]), mode = "image")
plt.at(1).look_at("xz")
plt.at(1).camera.Azimuth(180)
plt.at(1).roll(180)
plt.at(1).camera.Zoom(2)

temp_pos = plt.at(1).camera.GetPosition()

plt.at(1).camera.SetFocalPoint(cam_high[0], center[1], cam_high[2])
plt.at(1).camera.SetPosition(cam_high[0], temp_pos[1], cam_high[2])

plt.at(3).show(Picture("projector.png"),axes=0, zoom=1.5)

plt.at(2).show(Picture(f"result_text/{position_list[order - 1]}.png"),axes=0, zoom=1.9)

cam_distance = -temp_pos[1] - center[1]


font_color = "#f5f5f5"
normal_bg_color = "#424242"
hover_bg_color = "#43A047"

button_font = "FiraMonoMedium"
button_size = 35  
button_pos_spacing = 0.02 
button_y_pos = 0.02 


# Back Button
bu1 = plt.at(0).add_button(
    buttonfunc_back,
    pos=(0.05, button_y_pos),    # Adjusted x position
    states=["Back", "Back"],  # Using arrows for navigation buttons
    c=[font_color, font_color],  # Font color
    bc=[normal_bg_color, hover_bg_color],  # Background colors for each state (normal, hover)
    font=button_font,            # Font type
    size=button_size,            # Font size
    bold=True,                   # Bold font
    italic=False                 # Non-italic font style
)

# Next Button
bu2 = plt.at(0).add_button(
    buttonfunc_next,
    pos=(0.17, button_y_pos),    # Adjusted position for balance
    states=["Next", "Next"],  # Arrow for the next button
    c=[font_color, font_color],  # Font color
    bc=[normal_bg_color, hover_bg_color],  # Background colors
    font=button_font,            # Font type
    size=button_size,            # Font size
    bold=True,                   # Bold font
    italic=False                 # Non-italic font style
)

# NA Button
bu3 = plt.at(0).add_button(
    buttonfunc_na,
    pos=(0.29, button_y_pos),     # Adjusted position for alignment
    states=["N/A", "N/A"],        # Text for N/A
    c=[font_color, font_color],   # Font color
    bc=[normal_bg_color, hover_bg_color],  # Background colors
    font=button_font,             # Font type
    size=button_size,             # Font size
    bold=True,                    # Bold font
    italic=False                  # Non-italic font style
)

# Finish Button
bu4 = plt.at(0).add_button(
    buttonfunc_finish,
    pos=(0.41, button_y_pos),     # Adjusted position for final button
    states=["Finish", "Finish"],  # Text for Finish
    c=[font_color, font_color],   # Font color
    bc=[normal_bg_color, hover_bg_color],  # Background colors
    font=button_font,             # Font type
    size=button_size,             # Font size
    bold=True,                    # Bold font
    italic=False                  # Non-italic font style
)

plt.add_callback('ButtonLeftPress', func) # add the callback function
plt.at(3).remove_callback('mouse wheel')
plt.interactive().close()
