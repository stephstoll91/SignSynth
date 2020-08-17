import numpy as np
import matplotlib.pyplot as plt

#bc = [10, 30, 50, 80]
bc = [50, 80]

for bcc in bc:
    path = '/vol/vssp/smile/Steph/pycharm_projects/pose_regressor/results/g2p2v/' + str(bcc) + '_e5/eval/'
    err_e = []
    err_GT_e = []
    for e in range(1, 16):
        try:
            error_file = path + 'e' + str(e) + 'error.npz'
            err = np.load(error_file)
        except:
            error_file = path + 'e' + str(e-1) + 'error.npz'
            err = np.load(error_file)
        errs = err['errors_mse']
        errs_GT = err['errors_GT_mse']
        err_e.append(errs)
        err_GT_e.append(errs_GT)
        print("Epoch: " + str(e) + ", BCs: " + str(bcc) + ", MSE: " + str(errs) + "\n")


    #plt.plot([1, 2, 3, 4, 5], err_e, label='feedback_bc' + str(bcc))
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], err_e, label='feedback_bc' + str(bcc))
    #plt.plot([1, 2, 3, 4, 5], err_GT_e, label='from_GT_bc' + str(bcc))
plt.axis([1, 15, 0.0, 2.0])
#plt.axis([1, 5, 0.0, 1.5])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.savefig('/vol/vssp/smile/Steph/pycharm_projects/pose_regressor/50_80_MSEe15.png')
plt.close()