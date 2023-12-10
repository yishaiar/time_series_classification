import matplotlib.pyplot as plt
import pickle
def plot_accuarcy(str_):
    arr = pickle.load(open(str_, 'rb'))
    x,y,z = [],[],[]
    l = ['seq_len','accuarcy','epoch','batch_size','LEN']
    fig,ax  = plt.subplots(len(l[1:]),1,figsize=(10, 20))
    for j in range(len(l[1:])):
        for i in range(len(arr)):
            x.append(arr[i][0])
            y.append(arr[i][j+1])

        
            ax[j].plot(x,y,'.')
            ax[j].set_xlabel(l[0])
            ax[j].set_ylabel(l[j+1])
    plt.suptitle('accuarcy vs seq_len')
    plt.show()