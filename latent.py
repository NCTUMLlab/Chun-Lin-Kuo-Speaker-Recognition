import numpy as np
from scipy import io
import matplotlib.cm as cm



def plot_tsne(z_mu, classes, name):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_mu
    print(len(z_mu))
    # print(len(classes))
    # print(classes)
    print("tsne")
    z_embed = model_tsne.fit_transform(z_states)
    print("tsne....done")
    np.save('Gen_Data/z_embed_2.npy', z_embed)
    #print(z_embed)
    classes = classes
    #print("classes",classes)
    #print(len(classes))
    #print(len(z_embed))
    fig666 = plt.figure()
    count = np.zeros(np.max(classes))
    print("print the sorted class to class %d"%(count.shape[0]))
    for i in range(len(classes)):
        count[classes[i]-1] +=1
    print(count)
    for ic in range(len(classes)):
        #print(classes[ic])
        color = cm.rainbow(np.linspace(0, 1, np.max(classes)))
        #color = plt.cm.Set1(classes[ic])
        plt.scatter(z_embed[ic, 0], z_embed[ic, 1], s=30, color=color[classes[ic]-1])
        plt.title("2-D i-vector after GAN augumentation from class 1 to %d"%(count.shape[0]))
    fig666.savefig('./Latent_results/'+str(name)+'_embedding.png')


x =np.load('Gen_Data/gan_wc_x_latent2.npy')
y =np.load('Gen_Data/gan_wc_y_latent2.npy')
#print(x.shape)
num =100
x_sp = x[:num]
#print(x_sp.shape)
y_sp = y[:num]
#print(x_sp)

# print("y",y)
# print("x",x)
# X = io.loadmat('./save_mat/acgan_wc_x_5.mat')
# Y = io.loadmat('./save_mat/acgan_wc_y_5.mat')
plot_tsne(x_sp, y_sp, 'gan')