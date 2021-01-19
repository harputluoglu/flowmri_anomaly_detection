import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics

plt.figure()
plt.title("ROC CURVES")

noise_parameters = [(0.05, 0.05), (0.1, 0.1), (0.1, 0.3) , (0.3, 0.1)]

i = 0
colors = ['blue', 'purple', 'red', 'orange', 'magenta']

for (mean, stdev) in noise_parameters:
    tpr_file_path = '/scratch_net/biwidl210/peifferp/thesis/master-thesis/Results/Evaluation/Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled/AUC/TPR_NoiseMean_' + str(mean) + '_Std_' + str(stdev) + '.txt'
    fpr_file_path = '/scratch_net/biwidl210/peifferp/thesis/master-thesis/Results/Evaluation/Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled/AUC/FPR_NoiseMean_' + str(mean) + '_Std_' + str(stdev) + '.txt'

    tpr = np.genfromtxt(tpr_file_path)
    fpr = np.genfromtxt(fpr_file_path)

    auc_error_corr = 1. + np.trapz(fpr, tpr)

    print('Model: Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled || Noise Mean {}, Stdev {} --- AUC VALUE: {}'.format(model_name, str(mean), str(stdev), str(auc)))


    # Plotting
    tpr = tpr/tpr.max()
    fpr = fpr/fpr.max()

    plt.plot(fpr, tpr, color=colors[i], label="Noise: mean=" + str(mean) + ", stdev=" + str(stdev))

    i += 1


plt.legend()
plt.savefig('/scratch_net/biwidl210/peifferp/thesis/master-thesis/Results/Evaluation/Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled/AUC/ROC_CURVE.png)
