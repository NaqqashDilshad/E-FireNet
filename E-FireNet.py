import cv2
import glob
import numpy as np
import seaborn as sns
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report 


#======== Model Name, Input Size and Dataset Path ========#
from keras.applications.vgg16 import VGG16
inp = 128
model =  VGG16(weights='imagenet', include_top=False, input_shape=(inp, inp, 3))
model.summary()

model.layers.pop()
model = Model(model.input, model.layers[-4].output)
model.summary()

folders = glob.glob(r"D:\Code for Fire\MainDataset\*")

img_list = []
label_list=[]


#======== Dataset Resizing ========#
for folder in folders:
    print(folder) 
    for img in glob.glob(folder+r"/*.jpg"):
        n= cv2.imread(img)
        class_num = folders.index(folder)
        label_list.append(class_num)
        resized = cv2.resize(n, (inp,inp), interpolation = cv2.INTER_AREA)
        img_list.append(resized)


#======== Dataset Splitting ========#
X_train, X_test, y_train, y_test = train_test_split(img_list, label_list, test_size=0.1, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
X_test = np.array(X_test)
y_test = np.array(y_test)
print ("training_set", X_train.shape)
print ("training_set", y_train.shape)
print ("validation_set",X_valid.shape)
print ("validation_set",y_valid.shape)
print ("test_set",X_test.shape)
print ("test_set",y_test.shape)
print("Train_Folder",len(X_train))
print("validation_Folder",len(X_valid))
print("Test_Folder",len(X_test))

inputs = Input((inp, inp, 3))
X = model(inputs)
X = GlobalAveragePooling2D()(X)
X = Dense(3, activation='softmax')(X)
model = Model(inputs=inputs, outputs=X)
model.summary()


#======== Model Batch Size and Epochs ======== #
batch_size = 16
Epochs = 30
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X_train, y_train, batch_size=batch_size, epochs=Epochs,verbose=1,
                  validation_data=(X_valid,y_valid))


#======== Result Plotting ========#
f, ax = plt.subplots()
ax.plot([None] + history.history['accuracy'], 'o-')
ax.plot([None] + history.history['val_accuracy'], 'x-')
plt.savefig("D:\Code for Fire\Results\EdgeVGG16\EdgeVGG16_accuracy.png")


#======== Plot legend and use the best location automatically: loc = 0. ========#
ax.legend(['Train acc', 'Val acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
f, ax = plt.subplots()
ax.plot([None] + history.history['loss'], 'o-')
ax.plot([None] + history.history['val_loss'], 'x-')
plt.savefig("D:\Code for Fire\Results\EdgeVGG16\EdgeVGG16_loss.png")


#======== Plot legend and use the best location automatically: loc = 0. ========#
ax.legend(['Train loss', "Val loss"], loc = 1)
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()
print(history.history.keys())
plt.figure(figsize=(30,30))
sns.set(font_scale=1.0)

y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)

target_names = ["Car_Fire", "Building_Fire", "Non_Fire"] 
cm = confusion_matrix(y_test, y_pred)
print("***** Confusion Matrix *****")
print(cm)
print("***** Classification Report *****")
print(classification_report(y_test, y_pred, target_names=target_names))
classes=3


#======== Normalized Confusion Matrix ========#
#con = np.zeros((classes,classes))
#for x in range(classes):
#    for y in range(classes):
#        con[x,y] = cm[x,y]/np.sum(cm[x,:])


#======== Confusion Matrix Size and Color Customization ========# 
plt.figure(figsize=(10,8))
sns.set(font_scale=1.6) # for label size
df = sns.heatmap(cm, annot=True, fmt='g', cmap='Reds',
                 xticklabels= target_names , yticklabels= target_names)


#======== Normalized Confusion Matrix ========#
#df = sns.heatmap(con, annot=True,fmt='.2', cmap='Greens', 
               #xticklabels= target_names , yticklabels= target_names)
df.figure.savefig("D:\Code for Fire\Results\EdgeVGG16\EdgeVGG16.png")
plt.show()
model.save("D:\Code for Fire\Models\EdgeVGG16\EdgeVGG16.h5")


# =============================================================================
# print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'.format(*model.evaluate(X_test, y_test)))
# 
# print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
# 
# print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
# 
# print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
# 
# print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
# 
# cnf_matrix = confusion_matrix(y_test, y_pred)
# print(cnf_matrix)
# FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
# FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
# TP = np.diag(cnf_matrix)
# TN = cnf_matrix.sum()- (FP + FN + TP)
# 
# print ("My TN", TN)
# 
# #- (FP + FN + TP)
# tt=FP + FN + TP
# print ("My tt", tt)
# FP = FP.astype(float)
# FN = FN.astype(float)
# TP = TP.astype(float)
# TN = TN.astype(float)
# FPR=FP/(FP+TN)
# FNR=FN/(TP+FN)
# print("FPR\n", FPR*100,"\n")
# print("FNR", FNR*100)
# 
# FPR1=sum(FPR)
# print (FPR1)
# 
# # Fall out or false positive rate
# print ("False Positive Rate",sum(FPR)/len(FPR))
# # False negative rate
# print ("False Negative Rate",sum(FNR)/len(FNR))
# 
# cnf_matrix = confusion_matrix(y_test, y_pred)
# print(cnf_matrix)
# FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
# FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
# TP = np.diag(cnf_matrix)
# TN = cnf_matrix.sum() - (FP + FN + TP)
# FP = FP.astype(float)
# FN = FN.astype(float)
# TP = TP.astype(float)
# TN = TN.astype(float)
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
# # Overall accuracy
# ACC = (TP+TN)/cnf_matrix.sum()
# 
# print ("FPR\n",FPR)
# print ("Sensitivity or Recall\n",TPR)
# print ("Specificity \n",TNR)
# print ("Precision\n",PPV)
# print ("ACC\n",ACC)
# =============================================================================