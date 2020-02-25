from keras.models import Model, load_model
from keras.layers import Input, Dense
import pandas as pd
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt

df = pd.read_csv('Breast_cancer_data.csv')
input_tensor = Input(shape = (1,))
output_tensor_reg = Dense(1, activation = 'linear', use_bias= False)(input_tensor)
output_tensor_class = Dense(1, activation = 'sigmoid', use_bias= False)(output_tensor_reg)

model = Model(input_tensor,[output_tensor_reg,output_tensor_class])

model.compile(optimizer = Adam(0.01), loss = ['mae', 'binary_crossentropy'])

x = df['mean_area']
y_reg = df['mean_smoothness']
y_class = df['diagnosis']

model.fit(x,[y_reg,y_class],epochs = 10)

print(model.get_weights())
model.summary()
model.save('bc_model.h5')

df_1 =load_model('bc_model.h5')
df_1.summary()

print(df_1.evaluate(x,[y_reg,y_class], verbose = False))
plot_model(df_1, to_file = 'bb_model_2.png')
img = plt.imread('bb_model_2.png')
plt.imshow(img)
plt.show()


