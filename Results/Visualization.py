import numpy as np
import matplotlib.pyplot as plt

'''
mean_mae_1=np.loadtxt('mae_1.txt')
mean_mae_2=np.loadtxt('mae_2.txt')
epoch=50

plt.plot(range(1,epoch+1),mean_mae_1,'b',label='biased')
plt.plot(range(1,epoch+1),mean_mae_2,'r',label='unbiased')
plt.title('Validation mae per epoch')
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.legend()

plt.savefig("MAE.png")
plt.show()
'''
'''
fpr_1=np.loadtxt('fpr_1.txt')
tpr_1=np.loadtxt('tpr_1.txt')
fpr_2=np.loadtxt('fpr_2.txt')
tpr_2=np.loadtxt('tpr_2.txt')

plt.plot(fpr_1,tpr_1,'b',label='biased')
plt.plot(fpr_2,tpr_2,'r',label='unbiased')
plt.title('ROC')
plt.xlabel('False Positive Ratio')
plt.ylabel('True Positive Ratio')
plt.legend()

plt.savefig("ROC.png")
plt.show()
'''

mean_mae_a=np.loadtxt('mae_a.txt')
mean_mae_b=np.loadtxt('mae_b.txt')
epoch=20

plt.plot(range(1,epoch+1),mean_mae_a,'b',label='no regularization')
plt.plot(range(1,epoch+1),mean_mae_b,'r',label='regularization')
plt.title('Validation mae per epoch')
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.legend()

plt.savefig("MAE_1.png")
plt.show()