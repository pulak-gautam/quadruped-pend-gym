import xml.etree.ElementTree as ET
import math
import argparse
import numpy as np

from enum import Enum
from scipy.spatial.transform import Rotation
from colorama import Fore

def phi_F(x):
    return np.exp( - np.linalg.norm(x) / 0.25)

def mat2eulerScipy(mat):
    mat = mat.reshape((3,3))
    r =  Rotation.from_matrix(mat)
    euler = r.as_euler("zyx",degrees=True)
    return euler

def getAngle(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1.0:
        cos_theta = 1
    if cos_theta < -1.0:
        cos_theta = -1.0
    return np.arccos(cos_theta)

class StatusCodes(Enum):
    INFO = Fore.WHITE
    ERROR = Fore.RED
    WARNING = Fore.YELLOW

def display(status : str, data : str):
    try:
        print(f"{StatusCodes[status].value}[{StatusCodes[status].name}] {Fore.WHITE} {data}")
    except:
        print(f"{StatusCodes['ERROR'].value}[{StatusCodes['ERROR'].name}] Invalid status code {status} passed in display()!")

class XmlGenerator():
    """  XML generator given specifications of the pendulum

    Args:
        rho (float): Density of material (kg/m^3)
        height (float): height of cylindrical pendulum (m)
        radius (float): radius of cylindrical pendulum (m)
    """
    def __init__(self, 
                rho: float = 2710.0, 
                h: float = 1, 
                r: float = 0.01
        ):
        self.rho = rho
        self.h = h
        self.r = r
        self.m = self.rho * math.pi * self.r*self.r * self.h 
        self.MoI = [ (1/12) * self.m * self.h*self.h + 1/4 * self.m * self.r*self.r,  (1/12) * self.m * self.h*self.h + 1/4 * self.m * self.r*self.r,  (1/2) * self.m * self.r*self.r ]
        
    def run(self, 
            xml_path: str = "./quadruped_pend_gym/models/go2/go2.xml", 
            out_path: str = "./quadruped_pend_gym/models/go2/go2.xml"
        ):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        pole_body = root.find('.//body[@name="pole"]')

        if pole_body is not None:
            # Modify inertial parameters
            inertial = pole_body.find('inertial')
            if inertial is not None:
                inertial.set('mass', f'{self.m}')
                inertial.set('diaginertia', f'{self.MoI[0]} {self.MoI[1]} {self.MoI[2]}')
                inertial.set('pos', f'0 0 {self.h / 2}')
                inertial.set('quat', f'1 0 0 0')

            # Modify geom parameters
            geom = pole_body.find('geom')
            if geom is not None:
                geom.set('fromto', f'0 0 0 0.001 0 {self.h}')
                geom.set('size', f'{self.r}')

            comment = ET.Comment(f'Pendulum parameters height:{self.h}, radii:{self.r}, density:{self.rho}')
            display('INFO', f'Pendulum parameters height:{self.h}, radii:{self.r}, density:{self.rho}')

            pole_body.append(comment)
            tree.write(xml_path)
        else:
            display("ERROR", "Could not find body with name='pole' in XML.")



# Run with cli args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script to modify inverted pendulum inertial params given radii and height')

    # Default Parameters
    rho = 2710 #kg/m^3
    h = 1 #m
    r = 0.01 #m

    parser.add_argument('xml_file_path', type=str, help='Path to xml file of the model')
    parser.add_argument('--rho', type=float, default=rho, help=f'density in kg/m^3 (default = {rho})')
    parser.add_argument('--h', type=float, default=h, help=f'height in m (default = {h})')
    parser.add_argument('--r', type=float, default=r, help=f'radius in m (default = {r})')


    args = parser.parse_args()
    file_path = args.xml_file_path
    rho = args.rho
    h = args.h
    r = args.r

    m = rho * math.pi * r*r * h 
    MoI = [ (1/12) * m * h*h + 1/4 * m * r*r,  (1/12) * m * h*h + 1/4 * m * r*r,  (1/2) * m * r*r ]

    tree = ET.parse(file_path)
    root = tree.getroot()

    pole_body = root.find('.//body[@name="pole"]')

    if pole_body is not None:
        # Modify inertial parameters
        inertial = pole_body.find('inertial')
        if inertial is not None:
            inertial.set('mass', f'{m}')
            inertial.set('diaginertia', f'{MoI[0]} {MoI[1]} {MoI[2]}')
            inertial.set('pos', f'0 0 {h/2}')
            inertial.set('quat', f'1 0 0 0')

        # Modify geom parameters
        geom = pole_body.find('geom')
        if geom is not None:
            geom.set('fromto', f'0 0 0 0.001 0 {h}')
            geom.set('size', f'{r}')

        comment = ET.Comment(f'Pendulum parameters height:{h}, radii:{r}, density:{rho}')
        pole_body.append(comment)

        tree.write(file_path)
    else:
        print("Error: Could not find body with name='pole' in XML.")
