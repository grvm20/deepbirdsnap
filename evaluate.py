from keras.preprocessing.image import ImageDataGenerator
from inceptionv4 import create_model
from keras import metrics

data_directory = '../cropped_pred_scale1.2/'
#data_directory = '../cropped_aligned/'
train_data_dir = 'train'
validation_data_dir = 'validation'
test_data_dir = 'test'

model = create_model(num_classes=500, weights='best_weights/defrost_all_cropped_77.hdf5')
model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['accuracy', metrics.top_k_categorical_accuracy])


validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    data_directory+validation_data_dir,
    target_size=(299, 299),
    batch_size=100,
    class_mode='categorical')

validation_eval = []
count = 0

res = model.evaluate_generator(validation_generator, 3000//100)
print(res)

#for inp, label in validation_generator:
#    loss,accuracy = model.evaluate(inp, label, verbose=0)
#    validation_eval.append(accuracy)
#    count += len(inp)
#    print(count)

#validation_eval = np.mean(validation_eval, axis=0)
#print('Loss: {:.4f} Evaluate{:.4f}'.format(validation_eval[0], validation_eval[1]))
