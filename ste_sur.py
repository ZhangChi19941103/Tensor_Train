import tensorflow as tf

def layer(Inp,Inp_size,Neuron_num,lay_name,Act_function = None):
    with tf.name_scope(str(lay_name)):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([Inp_size,Neuron_num],dtype=tf.float32,seed=1),name='Weight')
            tf.summary.histogram('Weight',Weight)
        with tf.name_scope('Basic'):
            Basic = tf.Variable(tf.random_normal([400,Neuron_num],dtype=tf.float32,seed=1),name='Basic')
            tf.summary.histogram('Basic',Basic)
        with tf.name_scope('Out_Put'):
            Outputs_temp = tf.add(tf.matmul(Inp,Weight),Basic)

            if Act_function is None:
                Outputs = Outputs_temp
            else:
                Outputs = Act_function(Outputs_temp)
            tf.summary.histogram('Output',Outputs)

    return Outputs

def loss_step(Z_data,Z_pre):
    # 定义神经元损失函数并显示在tensorboard中，***reduction_indices***很重要
    # reduction_indices=[0]***********纵向压缩
    # reduction_indices=[1]***********横向压缩
    # reduction_indices=None**********总体压缩
    with tf.name_scope('Loss'):
        loss_pre = tf.reduce_mean(tf.reduce_sum(tf.square(Z_data-Z_pre),reduction_indices=[1]))
        tf.summary.scalar('Loss',loss_pre)
    return loss_pre

def train_step(loss):
    # 定义神经元训练函数并显示在tensorboard中
    # ******共11种优化器******
    with tf.name_scope('Train'):
        train = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)
    return train

def sess_step(mode = 1,proportion = 0.333):
    if mode == 1:
        #训练方式为指定使用一定比例的Gpu容量
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=proportion)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    elif mode == 2:
        #训练方式为按使用增长所占Gpu容量
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        #使用cpu训练模型
        sess = tf.Session()
    return sess

if __name__ == '__main__':
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    #构造Z_data
    Sh = 20
    x = np.linspace(-10,10,Sh,dtype=np.float32) + np.zeros((Sh,1),dtype=np.float32)
    y = np.linspace(-10,10,Sh,dtype=np.float32).reshape(Sh,1) + np.zeros((1,Sh),dtype=np.float32)
    noise = np.random.normal(0,0.5,size=(Sh,Sh)).astype(np.float32)
    # 拟合台阶面z_data
    z_temp = np.array(10*np.ones((Sh,Sh)),dtype=np.float32)
    z_temp[:,:int(Sh/2)] = 0
    z_data = z_temp.reshape(Sh*Sh,1)
    #可视化原始数据
    fig = plt.figure(0)
    ax = plt.subplot(111,projection = '3d')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(r'result/台阶面.avi', fourcc, 10, (2560, 1920))
    #构造数据矩阵
    inp_data = np.array(np.zeros((Sh*Sh,2)),dtype=np.float32)
    inp_data[:,0] = x.reshape((Sh*Sh,))
    inp_data[:,1] = y.reshape((Sh*Sh,))
    #构造Batch
    with tf.name_scope('Input_data'):
        Xs = tf.placeholder(tf.float32,[None,2],name='Xs')
        Zs = tf.placeholder(tf.float32,[None,1],name='Zs')
    #构造神经元，初始化损失函数与训练
    L1 = layer(Xs,2,10,lay_name='L_One',Act_function=tf.nn.relu)
    L2 = layer(L1,10,10,lay_name='L_Two',Act_function=tf.nn.relu)
    z_pre = layer(L2,10,1,lay_name='L_Out', Act_function=None)
    loss = loss_step(z_data,z_pre)
    train = train_step(loss)
    #开始训练
    with sess_step(mode=2) as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        Writer = tf.summary.FileWriter('log/ste_sur', sess.graph)
        los = 1
        i = 0
        #定义退出条件
        while los > 0.005 and i < 5000:
            sess.run(train,feed_dict={Xs:inp_data,Zs:z_data})
            los = sess.run(loss,feed_dict={Xs:inp_data,Zs:z_data})
            rs = sess.run(merged,feed_dict={Xs:inp_data,Zs:z_data})
            Writer.add_summary(rs)
            i  +=1
            if i < 100:
                try:
                    ax.cla()
                except Exception:
                    pass
                z_final = sess.run(z_pre, feed_dict={Xs: inp_data})
                lines = ax.plot_surface(x, y, z_final.reshape(Sh, Sh),cmap='rainbow')
                ax.scatter(x, y, z_data)
                plt.savefig(r'temp.jpg',dpi =400)
                img = cv2.imread(r'temp.jpg')
                videoWriter.write(img)
            elif i == 100:
                videoWriter.release()
            else:
                pass
            print(i,los)

        z_final = sess.run(z_pre, feed_dict={Xs: inp_data})
        ax.cla()
        ax.scatter(x, y, z_data.reshape(Sh, Sh))
        ax.plot_surface(x,y,z_final.reshape(Sh, Sh),cmap='rainbow')
        plt.savefig(r'result/temp.jpg', dpi=400)
        np.savetxt(r'result/data_st.csv', z_final.reshape(Sh, Sh), delimiter=',')
    os.remove('台阶面.jpg')

