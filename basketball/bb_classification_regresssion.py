from keras.layers import Input, Dense
import pandas as pd
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

df = pd.read_csv('games_tourney.csv')

input_tensor = Input(shape = (1,))
output_tensor_reg = Dense(1, activation = 'linear', use_bias= True)(input_tensor)
output_tensor_class = Dense(1, activation = 'sigmoid', use_bias = True)(output_tensor_reg)

model = Model(input_tensor,[output_tensor_reg, output_tensor_class])

model.compile(loss = ['mae','binary_crossentropy'],optimizer = Adam(0.01))

x = df[['seed_diff']]
y_reg = df['score_diff']
y_class = df['won']

model.fit(x,[y_reg,y_class], epochs = 1)
print(model.get_weights())
model.summary()
model.save('bb_model_1.h5')

df_1 =load_model('bb_model_1.h5')
df_1.summary()
#df_2 = pd.DataFrame(df_1)

#x_1 = df_2[['seed_diff']]
#y_reg_1 = df_2[['score_diff']]
#y_class_1 = df_2[['won']]

#df_2.fit(x_1,[y_reg_1, y_class_1], epochs = 1, verbose = True, batch_size = 16384)


print(df_1.evaluate(x,[y_reg,y_class], verbose = False))
print(sigmoid(1*0.23377538+0.01860397))
#pred = model.predict(x,[y_reg,y_class])
#print(pred)
plot_model(df_1, to_file = 'bb_model_2.png')
img = plt.imread('bb_model_2.png')
plt.imshow(img)
plt.show()
