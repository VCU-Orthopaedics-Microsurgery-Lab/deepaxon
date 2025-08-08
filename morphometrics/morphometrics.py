'''
-------------------------------- DEEPAXON --------------------------------
obtain morphometric data for a single segmented image file where the meylin is a middle grey and the axons are white
'''

import numpy as np
import cv2 #to work with images
from skimage.measure import label, regionprops_table #to get shape properties
import pandas as pd #make spreadsheet
from scipy import ndimage as ndi #for watershed segmentation
from skimage.feature import peak_local_max #for watershed segmentation
from skimage.segmentation import watershed #for watershed segmentation
import os

def get_labels(img):
    distance = ndi.distance_transform_edt(img)
    local_max_coords = peak_local_max(distance, min_distance=10)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = label(local_max_mask)
    segmented_cells = watershed(-distance, markers, mask=img)
    return segmented_cells

def get_myelin_row(myelin_df, x, y):
    '''
    Using the axon centroid coordinates, find the myelin that corresponds to that axon
    The centroid of the axon will be in the bbox of the myelin so find the proper myelin
    '''
    
    #make the axon centroid integers so they can be compared to bbox values
    x = int(x)
    y = int(y)
    
    #the axon centroid x will be left<x<right and the y will be top<y<bottom
    return myelin_df[((myelin_df['bbox-0'] <= x) & (myelin_df['bbox-2'] >= x)) & ((myelin_df['bbox-1'] <= y) & (myelin_df['bbox-3'] >= y))]

def get_axon_row(axon_df, left, right, top, bottom):
    '''
    Using the myelin bounding box find the larges axon that correspons to that myelin
    '''
    
    left = int(left)
    right = int(right)
    top = int(top)
    bottom = int(bottom)
    
    axons_id = axon_df[(((axon_df['centroid-0'])>=left) & (axon_df['centroid-0']<=right)) &
                       ((axon_df['centroid-1'])>=top) & (axon_df['centroid-1']<=bottom)]
    biggest_axon = axons_id[axons_id['area'] == axons_id['area'].max()]
    
    return biggest_axon
        
def get_morphometrics(img_path):
    '''
    Get morphometric data from a single image and return a pandas df of the morphometric data
    
    :param img_path: A path (string or object) pointing to a single segmented image in which the myelin is a middle grey and the axon is white
    
    :returns: Pandas DataFrame; morphometrics
    '''
    
    #read the image and flatten it into black and white
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    axon = cv2.inRange(img, 200, 255) #axons are everything above a medium gray
    myelin = cv2.inRange(img, 1, 255) #myelin is actually axon+myelin so everything above black
    
    #watershed labelling of axons then obtaining region properties and putting it in a DataFrame
    axon_label = get_labels(axon)
    axon_props = regionprops_table(axon_label,properties=('label', 'centroid', 'area', 'axis_minor_length', 'axis_major_length', 
                                                          'eccentricity', 'orientation', 'perimeter', 'solidity'))
    axon_df = pd.DataFrame.from_dict(axon_props)

    #watershed labelling of myelin then obtaining region properties and putting it in a DataFrame
    myelin_label = get_labels(myelin)
    myelin_props = regionprops_table(myelin_label, properties=('label', 'bbox', 'area', 'axis_minor_length', 'axis_major_length',
                                                               'perimeter'))
    myelin_df = pd.DataFrame.from_dict(myelin_props)
    
    #making an empty DataFrame to put morphometric data into
    columns = {'label':[],
               'x':[],
               'y':[],
               'axon_area':[],
               'axon_perimeter':[],
               'axon_diam':[],
               'myelin_area':[],
               'myelin_thickness':[],
               'myelin_perimeter':[],
               'eccentricity':[],
               'orientation':[],
               'solidity':[],
               'gratio':[]}
    morph_df1 = pd.DataFrame(columns)
    morph_df2 = pd.DataFrame(columns)
    
    #going through all the myelin that got accounted for in watershed
    #using myelin to compare to axons
    drop_rows = []    
    for index, row in myelin_df.iterrows():
        left = row['bbox-0']
        right = row['bbox-2']
        top = row['bbox-1']
        bottom = row['bbox-3']
        
        axon_row = get_axon_row(axon_df, left, right, top, bottom)
        if not(axon_row.empty):
            label = row['label']
            x = axon_row['centroid-0']
            y = axon_row['centroid-1']
            axon_area = axon_row['area']
            axon_perimeter = axon_row['perimeter']
            axon_diam = axon_row['axis_major_length']#(row['axis_major_length'] + row['axis_minor_length']) / 2 #getting axon diameter along the major axis
            eccentricity = axon_row['eccentricity']
            orientation = axon_row['orientation']
            solidity = axon_row['solidity']
            
            myelin_area = row['area'] - axon_area
            myelin_thickness = row['axis_major_length']
            myelin_perimeter = row['perimeter']
            gratio = axon_diam / row['axis_major_length']
            
            if gratio.iloc[0] < 1:
                #make a one row DataFrame with axon and myelin data
                new_dict = {'label':label,
                        'x':x,
                        'y':y,
                        'axon_area':axon_area,
                        'axon_perimeter':axon_perimeter,
                        'axon_diam':axon_diam,
                        'myelin_area':myelin_area,
                        'myelin_thickness':myelin_thickness,
                        'myelin_perimeter':myelin_perimeter,
                        'eccentricity':eccentricity,
                        'orientation':orientation,
                        'solidity':solidity,
                        'gratio':gratio}

                new_df = pd.DataFrame(new_dict)
                
                #add the new data to the morphometrics DataFrame
                morph_df2 = pd.concat([morph_df2,new_df], ignore_index=True)
            
    morph_df2 = morph_df2.drop(drop_rows)
    
    #going through all the axons that got accounted for in watershed
    #using axons since they are more separate so it will be a more accurate value
    # for index, row in axon_df.iterrows():
    #     #save all morphometric data in variables
    #     label = row['label']
    #     x = row['centroid-0']
    #     y = row['centroid-1']
    #     axon_area = row['area']
    #     axon_perimeter = row['perimeter']
    #     axon_diam = row['axis_major_length']#(row['axis_major_length'] + row['axis_minor_length']) / 2 #getting axon diameter along the major axis
    #     eccentricity = row['eccentricity']
    #     orientation = row['orientation']
    #     solidity = row['solidity']
    #     myelin_row = get_myelin_row(myelin_df, x, y) #get myelin corresponding to axon of interest
    #     myelin_area = myelin_row['area'] - axon_area #area of the myelin = (axon+myelin area) - (axon area)
    #     myelin_thickness = myelin_row['axis_major_length']#((myelin_row['axis_major_length'] + myelin_row['axis_minor_length']) / 2) - axon_diam #myelin thickness = (axon+myelin diam) - (axon diam)
    #     myelin_perimeter = myelin_row['perimeter']
    #     gratio = axon_diam / myelin_row['axis_major_length']#((myelin_row['axis_major_length'] + myelin_row['axis_minor_length']) / 2)
        
    #     #make a one row DataFrame with axon and myelin data
    #     new_dict = {'label':label,
    #             'x':x,
    #             'y':y,
    #             'axon_area':axon_area,
    #             'axon_perimeter':axon_perimeter,
    #             'axon_diam':axon_diam,
    #             'myelin_area':myelin_area,
    #             'myelin_thickness':myelin_thickness,
    #             'myelin_perimeter':myelin_perimeter,
    #             'eccentricity':eccentricity,
    #             'orientation':orientation,
    #             'solidity':solidity,
    #             'gratio':gratio}
    #     new_df = pd.DataFrame(new_dict)
        
    #     #add the new data to the morphometrics DataFrame
    #     morph_df1 = pd.concat([morph_df1,new_df], ignore_index=True)
        
    # return morph_df2

def save_morphometrics(morph_df, output_dir, output_name):
    output_path = os.path.join(output_dir, output_name+'.xlsx')
    morph_df.to_excel(output_path)