import pdb

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import numpy as np
import csv
import pdb
import pandas as pd
import os
#from matplotvideo import attach_video_player_to_figure


# filename = "D:\\Jakub work\\passing_to_other_hand\\passing_act1_6.csv"
filename = "test/A1s1c1-w.csv"
df = pd.read_csv(filename)
data = pd.DataFrame(df)

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

#data = data[:-226]
print('dropped bad columns and columns')
num_records, num_features = data.shape
print("there are {} flow records with {} feature dimension".format(num_records, num_features))
# print(data)


file = 'CSV_DATA.csv'
csv_data = data.to_csv(file)
# print(csv_data)

data = []
head = 0

with open(file, newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:

        if head == 0:
            head = 1
            continue
        # for meter
        
        data_line = [float(i) for i in row]

        
        # for milimeter
        # data_line = [float(i) for i in row]


        if data_line == []:
            # data.append([0] * 128)
            continue
        data.append(data_line)
        #pdb.set_trace()
# print(data)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(211, projection='3d')
ax.view_init(90, -90)
# ax.set_xlim(-1500, 1500)
# ax.set_ylim(-1000, 1000)
# ax.set_zlim(-1000, 1000)
# print(type(ax))

frame = range(num_records)

def update(frame):

    #plt.cla()
    ax.clear()
    frame1 = data[frame]
    # print(frame1)
    #ax.text(0.95, 0.95, f'Frame: {frame}', transform=ax.transAxes, ha='right', va='top')

    pelvis = [frame1[1], frame1[2], frame1[3]]
    stomach = [frame1[4], frame1[5], frame1[6]]
    chest = [frame1[7], frame1[8], frame1[9]]
    neck = [frame1[10], frame1[11], frame1[12]]
    Rshoulder = [frame1[13], frame1[14], frame1[15]]
    Rforearm = [frame1[16], frame1[17], frame1[18]]
    Relbow = [frame1[19], frame1[20], frame1[21]]
    Rwrist = [frame1[22], frame1[23], frame1[24]]
    Lshoulder = [frame1[25], frame1[26], frame1[27]]
    Lforearm = [frame1[28], frame1[29], frame1[30]]
    Lelbow = [frame1[31], frame1[32], frame1[33]]
    Lwrist = [frame1[34], frame1[35], frame1[36]]
    Rhip = [frame1[37], frame1[38], frame1[39]]
    Rknee = [frame1[40], frame1[41], frame1[42]]
    Rankle = [frame1[43], frame1[44], frame1[45]]
    Lhip = [frame1[46], frame1[47], frame1[48]]
    Lknee = [frame1[49], frame1[50], frame1[51]]
    Lankle = [frame1[52], frame1[53], frame1[54]]
    Rear = [frame1[55], frame1[56], frame1[57]]


    SkeletonConnectionMap = [[Rknee, Rankle],
                             [Rhip, Rknee],
                             [pelvis, Rhip],
                             [Lknee, Lankle],
                             [Lhip, Lknee],
                             [pelvis, Lhip],
                             [stomach, pelvis],
                             [chest, stomach],
                             [neck, chest],
                             [Relbow, Rwrist],
                             [Rforearm, Relbow],
                             [Rshoulder, Rforearm],
                             [neck, Rshoulder],
                             [Lelbow, Lwrist],
                             [Lforearm, Lelbow],
                             [Lshoulder, Lforearm],
                             [neck, Lshoulder],
                             [neck, Rear]]

    for joint_connection in SkeletonConnectionMap:
        endpoint_x = [joint_connection[0][0], joint_connection[1][0]]
        endpoint_y = [joint_connection[0][1], joint_connection[1][1]]
        endpoint_z = [joint_connection[0][2], joint_connection[1][2]]

        ax.plot(endpoint_x, endpoint_y, endpoint_z, c='b')
        ax.scatter(endpoint_x, endpoint_y, endpoint_z, c='b', marker='.')

    #plt.savefig(f'frame_{frame:03d}.png')

ani = FuncAnimation(fig, update, frames=num_records, interval=1000/15, repeat=False)
plt.tight_layout()

#ani = FuncAnimation(fig, update, frames=range(0, num_records), interval=1000/15, repeat=False)


#ani = FuncAnimation(fig, update, frames=range(0, num_records),
                    #init_func=init, blit=True)
#plt.show()

print("start recording")
#ani.save('C:\\Users\\User\\Desktop\\studies_materials\\Research_work\\Research_material_Copy\\pose_visualization\\pull_cart1.gif',
          #writer = 'pillow', fps = 15)
#ani.save('D:\\Jakub work\\vid5.gif', writer = 'pillow', fps = 15)

#print("done")

#image_file = 'representations/frames/frame_000.png'

# Read the image file
#img = mpimg.imread(image_file)

# Display the image
#plt.imshow(img)
plt.axis('off')  # Hide axis
plt.show()