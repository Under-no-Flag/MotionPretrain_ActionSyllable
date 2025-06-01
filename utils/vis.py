import os
import threading

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib import gridspec
from matplotlib import cm


# bone_pairs=[
#     (1, 0) ,(2, 1) ,(3, 2)  ,
#     (0, 4) ,(5, 4) ,(5, 6) ,
#     (0, 7) ,(7, 8) ,(8, 9) ,(9, 10) ,
#     (8, 14) ,(14, 15) ,(15, 16),
#     (8, 11) ,(11, 12) ,(12, 13) ,
# ]

# bone pairs for 17 key points
# bone_pairs=[
#     (0, 1) ,(1, 2) ,(2,3)  ,
#     (0, 4) ,(5, 4) ,(5, 6) ,
#     (8, 14) ,(14, 15) ,(15, 16),
#     (8, 11) ,(11, 12) ,(12, 13) ,
#     (0, 7) ,(7, 8) ,(8, 9) ,(9, 10) ,
# ]

# bone pairs for 22 key points
bone_pairs=[
    (0, 1) ,(1,2)  ,
    (4, 5) ,(5, 6) ,
    (8, 9), (9, 10), (10, 11),
    (9,12),(12, 13), (13, 14),
    (9, 17) ,(17, 18) ,(18, 19) ,
]



h36m_bone_pairs={
    'left_lower_limb':{
        'bones':[(0, 1) ,(1,2),(2,3)],
        'color': 'blue'
    },

    'right_lower_limb': {
        'bones': [(4, 5) ,(5, 6), (6, 7)],
        'color': 'green'
    },

    'right_upper_limb': {
        # 'bones': [(9, 17) ,(17, 18) ,(18, 19),(19,20),(19,21)],
        'bones': [(9, 17) ,(17, 18) ,(18, 19)],
        'color': 'green'
    },
    'left_upper_limb': {
        # 'bones': [(9,12),(12, 13), (13, 14),(14,16),(14,15) ],
        'bones': [(9,12),(12, 13), (13, 14)],
        'color': 'blue'
    },

    'backbone': {
        'bones': [(8, 9), (9, 10), (10, 11)],
        'color': 'green'
    },
    'hip': {
        'bones': [(-1, 0), (-1, 4), (-1, 8)],
        'color': 'green'
    },

}

actions = ["walking", "eating", "smoking", "discussion", "directions",
           "greeting", "phoning", "posing", "purchases", "sitting",
           "sittingdown", "takingphoto", "waiting", "walkingdog",
           "walkingtogether"]

# bone pairs for 22 key points
# bone_pairs=[
#     (1, 0) ,(2, 1) ,(3, 2),(3)
#     (0, 4) ,(5, 4) ,(5, 6) ,
#     (0, 7) ,(7, 8) ,(8, 9) ,(9, 10) ,
#     (8, 14) ,(14, 15) ,(15, 16),
#     (8, 11) ,(11, 12) ,(12, 13) ,
# ]


def perspective_projection(points, d=1):
    """
    透视投影，将3D坐标(x, y, z)投影到2D平面上。

    参数：
    - x, y, z: 坐标
    - d: 视点距离

    返回：
    - (xp, yp): 投影后的2D坐标
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 计算分母 (z + d)，并检查是否有零值以避免除以零
    denom = z + d
    if np.any(denom == 0):
        raise ValueError("z + d 不能为零，以避免除以零")

    # 计算投影后的二维坐标
    xp = (d * x) / denom
    yp = (d * y) / denom

    # 将 xp 和 yp 组合成一个 (V, 2) 的数组
    projected_points = np.stack((xp, yp), axis=1)

    return projected_points

def show3Dpose(skeleton_seq, ax, torques=None,radius=None,norm=None,cmap=None):
    sk=skeleton_seq


    # Plot the bones
    for part_name, body_part in h36m_bone_pairs.items():

        for bone in body_part['bones']:
            joint1, joint2 = bone

            # ax.plot(x,y,-z,color=body_part['color'])


            if torques is not None:
                # Get torque values for the two joints

                if part_name == 'hip':
                    torque1 = 0
                    torque2 = torques[joint2]
                    x = np.array([0, sk[joint2, 0]])
                    y = np.array([0, sk[joint2, 1]])
                    z = np.array([0, sk[joint2, 2]])
                else:
                    torque1 = torques[joint1]
                    torque2 = torques[joint2]
                    x = np.array([sk[joint1, 0], sk[joint2, 0]])
                    y = np.array([sk[joint1, 1], sk[joint2, 1]])
                    z = np.array([sk[joint1, 2], sk[joint2, 2]])


                # Interpolate colors based on torques
                color1 = cmap(norm(torque1))
                color2 = cmap(norm(torque2))

                # Create gradient colors for the line segment
                for i in range(10):
                    t = i / 9.0
                    interpolated_color = (1 - t) * np.array(color1) + t * np.array(color2)
                    ax.plot(
                        [x[0] * (1 - t) + x[1] * t, x[0] * (1 - (t + 1 / 9.0)) + x[1] * (t + 1 / 9.0)],
                        [y[0] * (1 - t) + y[1] * t, y[0] * (1 - (t + 1 / 9.0)) + y[1] * (t + 1 / 9.0)],
                        [-z[0] * (1 - t) - z[1] * t, -z[0] * (1 - (t + 1 / 9.0)) - z[1] * (t + 1 / 9.0)],
                        color=interpolated_color,
                        linewidth=2
                    )
            else:
                joint1, joint2 = bone
                if part_name == 'hip':
                    x = np.array([0, sk[joint2, 0]])
                    y = np.array([0, sk[joint2, 1]])
                    z = np.array([0, sk[joint2, 2]])
                    ax.plot(x, y, -z, color=body_part['color'])
                else:
                    x = np.array([sk[joint1, 0], sk[joint2, 0]])
                    y = np.array([sk[joint1, 1], sk[joint2, 1]])
                    z = np.array([sk[joint1, 2], sk[joint2, 2]])
                    ax.plot(x, y, -z, color=body_part['color'])

        # x,y,z=np.array([0,sk[0, 0]]),np.array([0,sk[0, 1]]),np.array([0,sk[0, 2]])
        # ax.plot(x, y, -z, color=h36m_bone_pairs['left_lower_limb']['color'])
        # x,y,z=np.array([0,sk[4, 0]]),np.array([0,sk[4, 1]]),np.array([0,sk[4, 2]])
        # ax.plot(x, y, -z, color=h36m_bone_pairs['right_lower_limb']['color'])
        # x,y,z=np.array([0,sk[8, 0]]),np.array([0,sk[8, 1]]),np.array([0,sk[8, 2]])
        # ax.plot(x, y, -z, color=h36m_bone_pairs['right_lower_limb']['color'])


    ax.grid(False)
    # ax.xaxis.line.set_visible(False)
    # ax.yaxis.line.set_visible(False)
    # ax.zaxis.line.set_visible(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axis('off')
    ax.view_init(elev=-75,azim=80,roll=15)
    pass


def show2Dpose(skeleton_seq, ax, offset):
    sk=perspective_projection(skeleton_seq)
    for _, body_part in h36m_bone_pairs.items():

        for bone in body_part['bones']:
            x = np.array([sk[bone[0],0]+offset[0], sk[bone[1],0]+offset[0]])
            y = np.array([sk[bone[0],1]+offset[1], sk[bone[1],1]+offset[1]])
            ax.plot(x,y,color=body_part['color'])
    return ax


def visualize_and_save_skeleton(
        skeleton_train_predict:list,
        epoch,
        batch_index,
        save_dir,
        subindex,
        prefix='train',
        torques=None,
        label=None,
):

    T=skeleton_train_predict[0].shape[0]
    num_list=skeleton_train_predict.__len__()


    # preprocessing torque first
    norm=None
    if torques is not None:
        torques = np.linalg.norm(torques, axis=-1)
        norm= Normalize(vmin=torques.min(), vmax=torques.max())
        cmap=cm.plasma
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    # plot bar
    if norm is not None:
        # fig, axs = plt.subplots(num_list, T+1, figsize=(T * num_list, 5), sharex=True, sharey=True,
        #                         subplot_kw={'projection': '3d'})

        fig, axs = plt.subplots(num_list, T + 1, figsize=(T * num_list, 5), sharex=True, sharey=True,
                                subplot_kw={'projection': '3d'})
        if num_list > 1:
            ax=axs[0,0]
        else:
            ax = axs[0]
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Torque Magnitude $(N \cdot m)$')
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.tick_left()
        cbar.ax.yaxis.set_ticks_position('right')
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.axis('off')
    else:
        fig, axs = plt.subplots(num_list, T, figsize=(T*num_list, 5), sharex=True, sharey=True,subplot_kw={'projection': '3d'})


    # plot skeleton
    for i,skeleton_seq in enumerate(skeleton_train_predict):

        for j,sk in enumerate(skeleton_seq):
            if norm is not None:
                if num_list>1:
                    show3Dpose(sk, axs[i,j+1], torques[j],radius=1,norm=norm,cmap=cmap)
                else:
                    show3Dpose(sk, axs[j+1], torques[j],radius=1,norm=norm,cmap=cmap)
            else:
                if num_list>1:
                    show3Dpose(sk, axs[i,j],radius=1)
                else:
                    show3Dpose(sk, axs[j],radius=1)




    plt.subplots_adjust(left=0.05, bottom=0.1, right=1, top=1, wspace=0, hspace=0.001)

    # plt.subplot_tool()
    if label is not None:
        plt.suptitle(f'{prefix} epoch {epoch} batch {batch_index} {actions[label]}')
        plt.savefig(f'{save_dir}/{prefix}_skeleton_epoch_{epoch}_batch{batch_index}_{subindex}_{actions[label]}.png',dpi=600)
    else:
        plt.savefig(f'{save_dir}/{prefix}_skeleton_epoch_{epoch}_batch{batch_index}_{subindex}.png',dpi=600)
    # plt.show()
    plt.close()


def visualize_2d(
        skeleton_train_predict:list,
        epoch,
        batch_index,
        save_dir,
        subindex,
        prefix='train'):

    T=skeleton_train_predict[0].shape[0]
    # gs=gridspec.GridSpec(2,T,hspace=0)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i,skeleton_seq in enumerate(skeleton_train_predict):

        for j,sk in enumerate(skeleton_seq):

            # ax=plt.subplot(int(f'2{T}{i*T+j+1}'),projection='3d')
            # ax=fig.add_subplot(skeleton_train_predict.__len__(),T,i*T+j+1,projection='3d')
            offset=[i*500,j*500]
            ax=show2Dpose(sk, ax, offset)

        # break


    # 绘制骨架的函数，这里只是示例，您需要根据骨架的具体格式来调整
    # fig.tight_layout()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=1, top=1, wspace=0, hspace=0.001)
    # plt.subplot_tool()
    plt.savefig(f'{save_dir}/{prefix}_skeleton_epoch_{epoch}_batch{batch_index}_{subindex}.png',dpi=600)
    # plt.show()
    plt.close()


# batch skeleton sequence
def plot_images(
        ground_truth,
        predict,
        batch_index,
        epoch,
        save_dir,
        show_length=10,
        num_sample=5
):

    N,T=ground_truth.shape[0],ground_truth.shape[2]
    sample_idx=np.linspace(0,N-1,num=num_sample).astype(int)
    for i in sample_idx:
        if ground_truth.shape[-1]!=3:
            ground_truth=ground_truth.transpose(0,2,3,1)
            predict=predict.transpose(0,2,3,1)
        show_idx=np.linspace(0, T-1, num=show_length).astype(int)
        visualize_and_save_skeleton([ground_truth[i][show_idx],predict[i][show_idx]],epoch,batch_index,save_dir,i)
        # visualize_and_save_skeleton([ground_truth[i][show_idx]],epoch,batch_index,save_dir,i)
    return None

def plot_skeleton_with_torque(
        ground_truth,
        torques,
        labels,
        thread_idx,
        save_dir,
        show_length=10,
        contiuum=True
):
    '''

    :param ground_truth: (N,T,V,C) or (N,C,T,V)
    :param torques: (N,T,V,C)
    :param save_dir:
    :param show_length:
    :return:
    '''
    if ground_truth.shape[-1]!=3:
        ground_truth=ground_truth.transpose(0,2,3,1)
    N,T=ground_truth.shape[0],ground_truth.shape[1]
    if contiuum:
        show_idx = np.arange(0, show_length).astype(int)
    else:
        show_idx = np.linspace(0, T - 1, num=show_length).astype(int)

    for i in range(N):
        visualize_and_save_skeleton([ground_truth[i][show_idx]],thread_idx,i,save_dir,0,prefix="train",torques=torques[i][show_idx],label=labels[i])
    return None

if __name__=='__main__':
    # npz_path = os.path.join('../data/h3.6mtxt', 'h36m_val_75.npz')
    # data = np.load(npz_path)
    # all_seqs, joint_used, dim_used, label_seqs = data['sampled_seq'], data['joint_to_use'], data['dimensions_to_use'], \
    #                                              data['label_seq']
    # sampled_torque_seq=data['sampled_torque_seq']
    # vis_T=5
    # vis_N=1
    # N,T=all_seqs.shape[0],all_seqs.shape[1]
    # all_seqs=all_seqs.reshape(N,T,-1,3)
    # skeleton=all_seqs[vis_N,30:vis_T+30,joint_used,:]
    # skeleton=skeleton.transpose(1,0,2)
    # torques=sampled_torque_seq[vis_N,30:vis_T+30,...]
    # visualize_and_save_skeleton([skeleton],0,0,save_dir='./',subindex=vis_N)





    '''
        vis all joints torques
    '''
    # read data from npz
    npz_path = os.path.join('../data/h3.6mtxt', 'h36m_val_75.npz')
    data = np.load(npz_path)
    save_dir='../vis_torque/h36m_new_torque_22'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_seqs, joint_used, dim_used, label_seqs = data['sampled_seq'], data['joint_to_use'], data['dimensions_to_use'], \
                                                    data['label_seq']

    sampled_torque_seq=data['sampled_torque_seq']
    vis_length=20
    N,T=all_seqs.shape[0],all_seqs.shape[1]
    all_seqs=all_seqs.reshape(N,T,-1,3)
    all_seqs=all_seqs[:,:,joint_used,:]


    # create multi-thread to visualize all joints and torques
    # each thread visualize m sequences
    # each sequence visualize vis_length frames
    # each frame visualize all joints and torques
    num_thread=1
    batch_size=N//num_thread
    thread_list=[]

    for i in range(num_thread):
        thread=threading.Thread(target=plot_skeleton_with_torque,args=(all_seqs[batch_size*i:batch_size*(i+1)],
                                                         sampled_torque_seq[batch_size*i:batch_size*(i+1)],
                                                         label_seqs[batch_size*i:batch_size*(i+1)],
                                                         i,
                                                         save_dir,
                                                         vis_length,
                                                         False))
        thread_list.append(thread)

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()


