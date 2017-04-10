from inceptionv4 import create_model

model = create_model()
print("Model created")
print(len(model.layers))


for layer in model.layers[::-1]:
    print (type(layer))
