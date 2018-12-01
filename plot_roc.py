import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load predictions
#y_actual = np.load(dhhd
# Load labels
#y_pred = np.load(dhfgs

y_actual = np.load("labels.np.npy")
y_pred = np.load("predictions.np.npy")

print(y_actual.shape)

y_pred = y_pred[:,:,0].reshape(1500)

print(y_pred.shape)
fpr, tpr, thresholds = roc_curve(y_actual, y_pred, pos_label = 1)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.save("fun")
