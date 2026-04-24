import pdb
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob as gb

# global variable
seed = 124
eps = 1e-15


# dataroot = "./activity2/"


# read all the csv files
def read_data(dataroot, file_ending='*.csv'):
    if file_ending is None:
        print("please specify file ending pattern for glob")
        exit()
    print(join(dataroot, file_ending))
    filenames = [i for i in gb.glob(join(dataroot, file_ending))]
    combined_csv = pd.concat([pd.read_csv(f, dtype=object) for f in filenames], sort=False,
                             ignore_index=True)  # dopisałem ignore_index=True
    return combined_csv  # returning a dataframe object


def read_single_data(datafile):  # datafile= address f file to be read
    out_csv = pd.read_csv(datafile)
    return out_csv


def transform_data(data1):
    data = data1.values
    # Find the indices of the rows containing the value x0
    #indices = np.where(data == a)[0]
    indices = [i for i, row in enumerate(data) if 'x0' in row]

    # Delete the rows containing the value x0
    data = np.delete(data, indices, axis=0)

    data = 1000 * data.astype(float)
    #data = data.astype(float)

    transformed_data = []
    for row in data:
        ref_point = [row[3], row[4], row[5]]
        i = 0
        subtracted = []
        while len(subtracted) < len(row):
            transf_point = row[i:i + 3]

            for item1, item2 in zip(ref_point, transf_point):
                subtracted.append(item2 - item1)

            i = i + 3
        transformed_data.append(subtracted)
    transformed_data = pd.DataFrame(transformed_data)
    return transformed_data



# load the csv files in data frame and start normalizing
def load_data_train1(datafile):
    # dataroot = "activity2"

    dataroot = datafile
    datas = []
    #data_paths = [read_data(dataroot, '*c1.csv'), read_data(dataroot, '*c2.csv')]  # list of length =2 containing camera 1 data (1st element) and camera 2 data(2ndd element)
    #data_paths = [read_data(dataroot, '*1-w.csv'), read_data(dataroot, '*2-w.csv')]
    data_paths = [read_data(dataroot, '*1_*'), read_data(dataroot, '*2_*'), read_data(dataroot, '*3_*'), read_data(dataroot, '*4_*'),
                    read_data(dataroot, '*5_*'), read_data(dataroot, '*6_*'), read_data(dataroot, '*7_*'), read_data(dataroot, '*8_*'), 
                    read_data(dataroot, '*9_*'), read_data(dataroot, '*10_*'), read_data(dataroot, '*11_*'), read_data(dataroot, '*12_*'),
                    read_data(dataroot, '*13_*'), read_data(dataroot, '*14_*'), read_data(dataroot, '*15_*'), read_data(dataroot, '*16_*'), 
                    read_data(dataroot, '*17_*'), read_data(dataroot, '*18_*'), read_data(dataroot, '*19_*'), read_data(dataroot, '*20_*'),
                    read_data(dataroot, '*21_*'), read_data(dataroot, '*22_*'),read_data(dataroot, '*23_*'),read_data(dataroot, '*24_*'),
                     read_data(dataroot, '*s1c*'), read_data(dataroot, '*s2*'), read_data(dataroot, '*s3*'), read_data(dataroot, '*s4*'), 
                  read_data(dataroot, '*s5*'), read_data(dataroot, '*s6*'), read_data(dataroot, '*s7*'), read_data(dataroot, '*s8*'),
                    read_data(dataroot, '*s9*'), read_data(dataroot, '*s10*'), read_data(dataroot, '*s11*'), read_data(dataroot, '*s12*')
                    
                ]
    
    for data_path in data_paths:
        num_records, num_features = data_path.shape
        print("there are {} flow records with {} feature dimension".format(num_records, num_features))

        # there is white spaces in columns names e.g. ' Destination Port'
        # So strip the whitespace from  column names
        data = data_path.rename(columns=lambda x: x.strip())
        print('stripped column names')

        
        # drop unnecessary coloum
        if data.shape[1] == 138:
            data = data.drop(
                columns=['x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10', 'x15', 'y15', 'z15', 'x16', 'y16', 'z16',
                        'x17', 'y17', 'z17', 'x21', 'y21', 'z21', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x28', 'y28',
                        'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30',
                        'x31', 'y31', 'z31', 'x32', 'y32', 'z32', 'x33', 'y33', 'z33'])
            data = data.drop(
                columns=['Frame number', 'person count', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                        'p11', 'p12', 'p13', 'p14',
                        'p15',
                        'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29',
                        'p30', 'p31', 'p32', 'p33'])

            for column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            
            data = data.dropna()
            
            """
            data = data.drop(
                columns = ['Frame number', 'person count', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10',
            'x15', 'y15',
            'z15', 'x16', 'y16', 'z16',
            'x17', 'y17', 'z17', 'x21', 'y21', 'z21', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x28', 'y28',
            'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30',
            'x31', 'y31', 'z31', 'x32', 'y32', 'z32', 'x33', 'y33', 'z33'])
            """
            
        if data.shape[1] == 128:
            data = data.drop(
                columns=['x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10',
                        'z10', 'x15', 'y15',
                        'z15', 'x16', 'y16', 'z16',
                        'x17', 'y17', 'z17', 'x21', 'y21', 'z21', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x28', 'y28',
                        'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30',
                        'x31', 'y31', 'z31'])

            data = data.drop(
                columns=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                        'p11', 'p12', 'p13', 'p14',
                        'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28',
                        'p29',
                        'p30', 'p31'])
            for column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            
            data = data.dropna()
            data.iloc[:, 1::3] = -data.iloc[:, 1::3]  #flip data accros y_axis (for Waseda data)
        

        print('dropped bad columns')
        num_records, num_features = data.shape
        print("there are {} flow records with {} feature dimension".format(num_records, num_features))


        data = transform_data(data)



        nan_count = data.isnull().sum().sum()
        print('There are {} nan entries'.format(nan_count))

        if nan_count > 0:
            data.fillna(data.mean(), inplace=True)
            print('filled NAN')
        if nan_count > 0:  # if still there are nones
            print("\nReplacing NaNs with the value from the previous row:")
            data.fillna(method='pad', inplace=True)
        if nan_count > 0:
            print(data)
            print("\nReplacing NaNs with the value from the next row:")
            data.fillna(method='bfill', inplace=True)
            # Normalising all numerical features:
        # cols_to_norm = list(data.columns.values)[:68]
        # print('cols_to_norm:\n', cols_to_norm)
        data = data.astype(np.float32)

        #mask = data == -1
        #data[mask] = 0  # replace what is -1 by 0

        # to leave -1 (missing features) values as is and exclude in normilizing
        #mean_i = np.mean(data, axis=0)
        #min_i = np.min(data, axis=0)
        #max_i = np.max(data, axis=0)
        # zero centered
       ##data = (data - mean_i) / r

        # deal with missing features -1
        # data[mask] = 0

        data = data.astype(float).apply(pd.to_numeric)

        print('converted to numeric\n', data)
        data = data.values
        data = data.tolist()
        # data.hist(figsize=(3,5))
        # plt.show()


        datas.extend(data)
    return datas



def load_data_test1(datafile):  # take in .csv folder address
    # dataroot = "test_activity1"
    dataroot = datafile
    datas = []
    data_paths = read_data(dataroot,
                           '*.csv')  # list of length =2 containing camera 1 data (1st element) and camera 2 data(2ndd element)

    num_records, num_features = data_paths.shape
    print("there are {} flow records with {} feature dimension".format(num_records, num_features))

    # there is white spaces in columns names e.g. ' Destination Port'
    # So strip the whitespace from  column names
    data = data_paths.rename(columns=lambda x: x.strip())
    #print('stripped column names')

    # drop unnecessary coloumn
    
    if data.shape[1] == 138:
        data = data.drop(
            columns=['x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10', 'x15', 'y15', 'z15', 'x16', 'y16', 'z16',
                    'x17', 'y17', 'z17', 'x21', 'y21', 'z21', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x28', 'y28',
                    'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30',
                    'x31', 'y31', 'z31', 'x32', 'y32', 'z32', 'x33', 'y33', 'z33'])
        data = data.drop(
            columns=['Frame number', 'person count', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                    'p11', 'p12', 'p13', 'p14',
                    'p15',
                    'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29',
                    'p30', 'p31', 'p32', 'p33'])
        
        for column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            
        data = data.dropna()


    if data.shape[1] == 128:
        data = data.drop(
            columns=['x1', 'y1', 'z1', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10',
                        'z10', 'x15', 'y15',
                        'z15', 'x16', 'y16', 'z16',
                        'x17', 'y17', 'z17', 'x21', 'y21', 'z21', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x28', 'y28',
                        'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30',
                        'x31', 'y31', 'z31'])

        data = data.drop(
            columns=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                        'p11', 'p12', 'p13', 'p14',
                        'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28',
                        'p29',
                        'p30', 'p31'])
        for column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            
        data = data.dropna()
        data.iloc[:, 1::3] = -data.iloc[:, 1::3]  #flip data accros y_axis (for Waseda data)
            


    #data = data.drop(
            #columns=['Frame number', 'person count', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10',
                    # 'z10',
                    # 'x15', 'y15',
                    # 'z15', 'x16', 'y16', 'z16',
                     #'x17', 'y17', 'z17', 'x21', 'y21', 'z21', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x28', 'y28',
                    # 'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30',
                    # 'x31', 'y31', 'z31', 'x32', 'y32', 'z32', 'x33', 'y33', 'z33'])


    print('dropped bad columns')
    num_records, num_features = data.shape
    print("there are {} flow records with {} feature dimension".format(num_records, num_features))

    #data = transform_data(data)

    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count > 0:
        data.fillna(data.mean(), inplace=True)
        print('filled NAN')
    if nan_count > 0:  # if still there are nones
        print("\nReplacing NaNs with the value from the previous row:")
        data.fillna(method='pad', inplace=True)
    if nan_count > 0:
        print(data)
        print("\nReplacing NaNs with the value from the next row:")
        data.fillna(method='bfill', inplace=True)
        # Normalising all numerical features:
    # cols_to_norm = list(data.columns.values)[:68]
    # print('cols_to_norm:\n', cols_to_norm)
    data = data.astype(np.float32)

    #mask = data == -1
    #data[mask] = 0  # replace what is -1 by 0

    # to leave -1 (missing features) values as is and exclude in normilizing
    #mean_i = np.mean(data, axis=0)
    #min_i = np.min(data, axis=0)
    #max_i = np.max(data, axis=0)
    # zero centered
    #r = (max_i - min_i) + eps
    #data = (data - mean_i) / r

    # deal with missing features -1
    # data[mask] = 0

    data = data.astype(float).apply(pd.to_numeric)

    print('converted to numeric\n', data)
    # data.hist(figsize=(3,5))
    # plt.show()
    data = data.values
    data = data.tolist()
    # data.hist(figsize=(3,5))
    # plt.show()

    datas.extend(data)
    return datas


#h= load_data_train1("all_activities_train")
#h= load_data_train1("20_activities_train")
#print("length is ", len(h))

# print(h)
