import numpy as np
import matplotlib.pyplot as plt

bc = [10, 30, 50, 80]
#bc = [50]
for bcc in bc:
    path = '/home/steph/Documents/Chapter2/Code/smile-tmp/test/new_normed_BC' + str(bcc) + '_h128_no_eval/'
    #path = '/home/steph/Documents/Chapter2/Code/smile-tmp/test/BC' + str(bcc) + '_h128_e50/'
    err_e = []
    err_GT_e = []
    for e in range(2, 22):
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
    plt.plot(np.arange(1, 21), err_e, label='feedback_bc' + str(bcc))
    plt.plot(np.arange(1, 21), err_GT_e, label='fromGT_bc' + str(bcc))
    #plt.plot([1, 2, 3, 4, 5], err_GT_e, label='from_GT_bc' + str(bcc))
plt.axis([1, 20, 0.0, 12.0])
plt.xticks([5, 10, 15, 20])
#plt.axis([1, 5, 0.0, 1.5])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.close()

