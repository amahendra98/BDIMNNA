"""
This is the helper functions for evaluation purposes

"""
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils.helper_functions import simulator


def get_test_ratio_helper(flags):
    """
    The unified place for getting the test_ratio the same for all methods for the dataset,
    This is for easier changing for multi_eval
    """
    if flags.data_set == 'ballistics':
        #return 0.00781                       # 100 in total
        return 0.039                        # 500 in total
    elif flags.data_set == 'sine_wave':
        #return 0.0125                        # 100 in total
        return 0.0625                        # 500 in total
    elif flags.data_set == 'robotic_arm':
        return 0.05                          # 500 in total
        #return 0.01                          # 100 in total
    elif flags.data_set == 'meta_material':
        return 0.02                        # 10000 in total for Meta material
    else:
        print("Your dataset is none of the artificial datasets")
        return None

def compare_truth_pred(pred_file, truth_file, cut_off_outlier_thres=None, quiet_mode=False):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    if isinstance(pred_file, str):      # If input is a file name (original set up)
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
    elif isinstance(pred_file, np.ndarray):
        pred = pred_file
        truth = truth_file
    else:
        print('In the compare_truth_pred function, your input pred and truth is neither a file nor a numpy array')
    if not quiet_mode:
        print("in compare truth pred function in eval_help package, your shape of pred file is", np.shape(pred))
    if len(np.shape(pred)) == 1:
        # Due to Ballistics dataset gives some non-real results (labelled -999)
        valid_index = pred != -999
        if (np.sum(valid_index) != len(valid_index)) and not quiet_mode:
            print("Your dataset should be ballistics and there are non-valid points in your prediction!")
            print('number of non-valid points is {}'.format(len(valid_index) - np.sum(valid_index)))
        pred = pred[valid_index]
        truth = truth[valid_index]
        # This is for the edge case of ballistic, where y value is 1 dimensional which cause dimension problem
        pred = np.reshape(pred, [-1,1])
        truth = np.reshape(truth, [-1,1])
    mre = np.mean(np.abs((pred-truth)/truth), axis=1)
    mae = np.mean(np.abs(pred-truth), axis=1)
    mse = np.mean(np.square(pred-truth), axis=1)

    if cut_off_outlier_thres is not None:
        mre = mre[mre < cut_off_outlier_thres]
        mse = mse[mse < cut_off_outlier_thres]
        mae = mae[mae < cut_off_outlier_thres]

    return mae, mre, mse

def sampleSpectra(n,targets, pred_file, truth_file, MRE=False, MAE=False, quantiles=True):
    if isinstance(pred_file, str):      # If input is a file name (original set up)
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
    elif isinstance(pred_file, np.ndarray):
        pred = pred_file
        truth = truth_file
    else:
        raise Exception('In the findSpectra function, your input pred and truth is neither a file nor a numpy array')

    mae,mre,mse = compare_truth_pred(pred,truth)

    if MRE:
        # search by MRE
        sequence = mre
    elif MAE:
        # search by MAE
        sequence = mae
    else:
        # search by MSE default
        sequence = mse

    # Targets are quantile values
    quant_vals = targets
    if quantiles:
        quant_vals = [np.quantile(sequence, q) for q in targets]

    ret_list = []

    for q,trgt in zip(quant_vals,targets):
        distance = np.abs(sequence-q)
        sorter = np.argsort(np.argsort(distance))
        idcs = np.argwhere(sorter < n)
        ret_list.append({'target':trgt,'idxs':np.reshape(idcs,n),'mse':np.reshape(mse[idcs],n),
                         'mre':np.reshape(mre[idcs],n),'mae':np.reshape(mae[idcs],n),'pred': np.squeeze(pred[idcs,:]),
                         'truth':np.squeeze(truth[idcs,:])})
    return ret_list

def plotSpectra(x_range,pred_spec,truth_spec,labels, title,save_str='data/temp'):
    f = plt.figure()
    colors = ['c','g','y','r','k','m','b']
    random.shuffle(colors)
    for pred,truth,label in iter(zip(pred_spec,truth_spec,labels)):
        c = colors.pop()
        plt.plot(x_range, truth, '-.' + c)
        plt.plot(x_range, pred, '-' + c, label=label)
    plt.xlabel("Wavelength")
    plt.ylabel("Spectra")
    plt.legend(loc="upper left")
    plt.suptitle(title)
    plt.savefig('{}.png'.format(save_str))

def plotMSELossDistrib(pred_file, truth_file, flags, save_dir='data/',quantiles=None):
    if (flags.data_set == 'gaussian_mixture'): #data_set
        # get the prediction and truth array
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
        # get confusion matrix
        cm = confusion_matrix(truth, pred)
        cm = cm / np.sum(cm)
        # Calculate the accuracy
        accuracy = 0
        for i in range(len(cm)):
            accuracy += cm[i,i]
        print("confusion matrix is", cm)
        # Plotting the confusion heatmap
        f = plt.figure(figsize=[15,15])
        plt.title('accuracy = {}'.format(accuracy))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True)
        eval_model_str = flags.eval_model.replace('/','_') #eval_model
        f.savefig(save_dir + '{}.png'.format(eval_model_str),annot_kws={"size": 16})

    else:
        mae, mre, mse = compare_truth_pred(pred_file, truth_file)
        plt.figure(figsize=(12, 6))
        y,x,_ = plt.hist(mse, bins=100)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:.4e}, Avg MRE={:.4%})'.format(np.mean(mse),np.mean(mre)))
        eval_model_str = flags.eval_model.replace('/','_')

        if quantiles:
            qts = [np.quantile(mse, q) for q in quantiles]
            for i in range(5):
                plt.axvline(qts[i], ymax = y.max(), linestyle = ":")
                plt.text(qts[i],y.max()*(10-i)/10,"{}th".format(quantiles[i]*100),fontsize=12)

        plt.savefig(os.path.join(save_dir,'{}.png'.format(eval_model_str)))
        print('(Avg MSE={:.4e}, Avg MRE={:.4%})'.format(np.mean(mse),np.mean(mre)))

def makePlots(pred_file,truth_file,flags,save_dir='data/',quantiles=None):
    # Create the histogram
    plotMSELossDistrib(pred_file,truth_file,flags,save_dir=save_dir,quantiles=quantiles)

    # Sample desired spectra from list
    samples = sampleSpectra(3,quantiles,pred_file,truth_file,quantiles=True)

    if flags.data_set == "peurifoy":
        x_range = np.linspace(400,800,len(samples[0]['pred'][0]))
    elif flags.data_set == "chen":
        x_range = np.linspace(240,2000,len(samples[0]['pred'][0]))
    elif flags.data_set == "sine_wave":
        x_range = np.linspace(0,1,len(samples[0]['pred'][0]))
    elif flags.data_set == "meta_material":
        x_range = np.linspace(0.5,2,len(samples[0]['pred'][0]))
    elif flags.data_set == "ballistics":
        x_range = np.linspace(0,1,len(samples[0]['pred'][0]))
    elif flags.data_set == "robotic_arm":
        x_range = np.linspace(0,1,len(samples[0]['pred'][0]))
    else:
        raise Exception("Conigure data_set option in flags object")

    # Create plots
    for i,plot in enumerate(samples):
        label = []
        for j in range(len(plot['idxs'])):
            label.append('MSE: {}, MRE: {}'.format(format(plot['mse'][j],'.4g'), format(plot['mre'][j],'.2%')))

        if quantiles:
            title = "Spectral fits at {}th quantile".format(int(100*plot['target']))
            save = os.path.join(save_dir,flags.eval_model+'_'+str(int(100*plot['target']))+'th')
        else:
            title = "Spectral fits at {} value".format(plot['target'])
            save = os.path.join(save_dir, flags.eval_model+'_val'+str(plot['target']))

        plotSpectra(x_range,plot['pred'],plot['truth'],label,title,save_str=save)



def eval_from_simulator(Xpred_file, flags):
    """
    Evaluate using simulators from pred_file and return a new file with simulator results
    :param Xpred_file: The prediction file with the Xpred in its name
    :param data_set: The name of the dataset
    """
    Xpred = np.loadtxt(Xpred_file, delimiter=' ')
    Ypred = simulator(flags.data_set, Xpred)
    Ypred_file = Xpred_file.replace('Xpred', 'Ypred_Simulated')
    np.savetxt(Ypred_file, Ypred)
    Ytruth_file = Xpred_file.replace('Xpred','Ytruth')
    plotMSELossDistrib(Ypred_file, Ytruth_file, flags)
