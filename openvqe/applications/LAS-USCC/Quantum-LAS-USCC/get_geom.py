import numpy as np

def sph2cart(az, el, r=0.5123473957082245):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def get_geom (geom_type, dist = 0.0, angles =[-0.3031323952381563, -0.1964410468213601]):

    if geom_type == 'close':
        xyz = '''H 0.0 0.0 0.0; 
                 H 1.0 0.0 0.0;
                 H 0.2 1.6 0.1;
                 H 1.159166 1.3 -0.1'''

    elif geom_type == "far":
        xyz = '''H 0.0 0.0 0.0;
                 H 1.0 0.0 0.0;
                 H 0.2 3.9 0.1;
                 H 1.159166 4.1 -0.1'''

    elif geom_type == "scan":
        mid = np.array([0.12291127, 0.992417664 , 0.0]) * (1.461078387+dist)
        mid[0] += 0.5
        disp = np.array(sph2cart( angles[0], angles[1]))
        h3=mid-disp
        h4=mid+disp
        xyz = '''H 0.0 0.0 0.0;
                 H 1.0 0.0 0.0;
                 H {} {} {};
                 H {} {} {}'''.format(h3[0],h3[1],h3[2], h4[0],h4[1],h4[2])
    else:
        raise Exception('Was that a typo??  Pick only one of close, far or scan and include parameters for scan otherwise you will get the close geometry')

    return xyz


