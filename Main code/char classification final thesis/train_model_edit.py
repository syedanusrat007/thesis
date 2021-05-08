from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from net import Net # creating modelNameName

wdthIm,hghtIm = 32,32 # set image size
channelNo=3 # set image number
data_dir_train='test_together/'
data_dirc_validation = 'test_together/'
epochNo=120
batchSize=52

modelName=Net.build(width = wdthIm,height = hghtIm,depth = channelNo) #modelName name
print('building done!!')

Rms=optimizers.Rmsprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0) #compiling model
print('optimizing done!!')
modelName.compile(loss='categorical_crossentropy',
              optimizer= Rms,
              metrics= ['accuracy'])

print('compiling.....')

train_datagenerator=ImageDataGenerator(
    featurewise_centering = True,
    featurewise_std_normaliztn = True,
	shear_ranges= 0.1,
    zoom_ranges =0.1,
    rotation_ranges= 5,
	height_shift_ranges= 0.05,
    width_shift_ranges= 0.05,
    horizontal_fliping= False,
	rescaling= 1. / 255,)
test_datageneratorerator = ImageDataGenerator(featurewise_centering= True, rescaling= 1. / 255, featurewise_std_normaliztn= True,)

train_generators  =train_datagenerator.flow_from_directory(
    data_dir_train,
	batchSize=batchSize,
    target_size=(wdthIm, hghtIm),
    class_mode='categorical')

validation_generators =test_datageneratorerator.flow_from_directory(
    data_dirc_validation,
    target_size=(wdthIm, hghtIm),
    batchSize=batchSize,
    class_mode='categorical')

history =modelName.fit_generator(
    train_generators,
    steps_per_epoch= train_generators.samples/batchSize,
    validation_data= validation_generators,
	epochNo= epochNo,
    validation_steps= validation_generators.samples/batchSize)

modelName.evaluate_generator(validation_generators)
modelName.save_weights('trained_weights.h5')
modelName.summary()
print(history.history)


plt.figure(figsize=[8,7])  #PLOT LOSS
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.legend(['Training loss...', 'Validation Loss...'],fontsize=18)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.xlabel('epochNo ',fontsize=16)
plt.ylabel('Loss ',fontsize=16)
plt.title('Loss_Curves ',fontsize=16)


# PLOT ACCURACY
plt.figure(figsize=[8,7])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy...', 'Validation Accuracy...'],fontsize=18)
plt.xlabel('epochNo ',fontsize=16)
plt.ylabel('Accuracy ',fontsize=16)
plt.title('Accuracy_Curves ',fontsize=16)
plt.show()



