import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    ip = np.ones(shape=(100, 30))
    op = np.ones(shape=(100,))

    mip = tf.keras.layers.Input(shape=(30,))

    mip2 = tf.expand_dims(mip, axis=-1)
    h = tf.keras.layers.Dense(units=5, activation='tanh')(mip2)
    s = tf.keras.layers.Dense(units=1, activation='tanh')(h)
    s = tf.keras.layers.Flatten()(s)
    w = tf.math.sigmoid(s)
    model_input = mip * w

    h1 = tf.keras.layers.Dense(units=100, activation='relu')(model_input)
    h2 = tf.keras.layers.Dense(units=50, activation='relu')(h1)
    mop = tf.keras.layers.Dense(units=1, activation='softmax')(h2)
    model = tf.keras.models.Model(inputs=mip, outputs=mop)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(ip, op)
    model.summary()

    get_w = tf.keras.models.Sequential(
        *model.layers[:6]
    )
    get_w.predict(ip)
    print(1)



