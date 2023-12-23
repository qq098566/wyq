## KPB: Attention U-Net for accurate radiotherapy dose prediction 
# @author: Alexander F.I. Osman, April 2021

"""
This code demonstrates an attention U-Net model for voxel-wise dose prediction in radiation therapy.
The model is trained, validated, and tested using the OpenKBP—2020 AAPM Grand Challenge dataset.
"""

##########################################################

# Import libraries
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import random
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error

def coup_nitrogen():
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    ###############################################################################
    # 1. LOAD DATA AND PREFORM PRE-PROCESSING #####################################
    ###############################################################################
    # Parameters for model
    mode = 'predict' # 'train' 'predict'
    model_name = 'UNet_best_model.epoch175-loss0.00014.hdf5'
    path = 'coup_nitrogen'
    img_height = 128
    img_width = 128
    img_depth = 128
    img_channels = 4
    input_shape = (img_height, img_width, img_depth, img_channels)

    dose_type = 'nitrogen'

    epochs = 300
    batch_size = 3
    steps_per_epoch = 90 // batch_size
    val_steps_per_epoch = 9 // batch_size

    # 定义数据加载函数
    def data_load(dose_type, start_number, end_number):
        new_dir = '../../PHITS_data_processed_for_3DUnet_z_up/4_ratio'
        stan_dir = '../../PHITS_data_processed_for_3DUnet_z_up/1_ratio'
        x = np.zeros((end_number-start_number, 128, 128, 128, 4))
        y = np.zeros((end_number-start_number, 128, 128, 128, 1))
        for i in range(start_number, end_number):
            patient_number = str(i).zfill(3)
            save_path = os.path.join(new_dir, f'GLI_{patient_number}_GBM_dose_data', 'data')
            detail_dose = np.load(os.path.join(save_path, dose_type + 'detail_doses.npy'))

            x[i - start_number, :, :, :, 0] = np.transpose(detail_dose, (1, 2, 0))
            x[i - start_number, :, :, :, 1] = np.transpose(np.load(os.path.join(save_path, 'hus.npy')), (1, 2, 0))
            x[i - start_number, :, :, :, 2] = np.transpose(np.load(os.path.join(save_path, 'bodies.npy')), (1, 2, 0))
            x[i - start_number, :, :, :, 3] = np.transpose(np.load(os.path.join(save_path, 'skins.npy')),(1, 2, 0))
            # x[i - start_number, :, :, :, 4] = np.transpose(np.load(os.path.join(save_path, 'angle.npy')), (1, 2, 0))

            stan_path = os.path.join(stan_dir, f'GLI_{patient_number}_GBM_dose_data', 'data')
            y[i - start_number, :, :, :, 0] = np.transpose(np.load(os.path.join(stan_path, dose_type + 'detail_doses.npy')), (1, 2, 0))

            # rows = np.transpose(np.load(os.path.join(save_path, dose_type + 'rows.npy')), (1, 2, 0)) * 128
            # cols = np.transpose(np.load(os.path.join(save_path, dose_type + 'cols.npy')), (1, 2, 0)) * 128
            # slices = np.transpose(np.load(os.path.join(save_path, dose_type + 'slices.npy')), (1, 2, 0)) * 128

            # y_height_sum_2d = np.sum(rows, axis=0)
            # y_height_sum_1d = np.sum(cols, axis=(1, 2))
            # y_height_sum_1d_ratio =  y_height_sum_1d / np.sum(y_height_sum_1d)
            # for a in range(y[i - start_number, :, :, :, 0].shape[0]):
            #     x[i - start_number, a, :, :, 0] = y_height_sum_2d*y_height_sum_1d_ratio[a]
            #
            # y_weight_sum_2d = np.sum(cols, axis=1)
            # y_weight_sum_1d = np.sum(slices, axis=(0, 2))
            # y_weight_sum_1d_ratio = y_weight_sum_1d / np.sum(y_weight_sum_1d)
            # for a in range(y[i - start_number, :, :, :, 0].shape[1]):
            #     x[i - start_number, :, a, :, 1] = y_weight_sum_2d * y_weight_sum_1d_ratio[a]
            #
            # y_depth_sum_2d = np.sum(slices, axis=2)
            # y_depth_sum_1d = np.sum(rows, axis=(0, 1))
            # y_depth_sum_1d_ratio = y_depth_sum_1d / np.sum(y_depth_sum_1d)
            # for a in range(y[i - start_number, :, :, :, 0].shape[2]):
            #     x[i - start_number, :, :, a, 2] = y_depth_sum_2d * y_depth_sum_1d_ratio[a]

        return x, y

    # region
    # region
    ###############################################################################
    # 2. BUILD THE MODEL ARCHITECTURE #############################################
    ###############################################################################
    # For consistency
    # Since the neural network starts with random initial weights, the results of this
    # example will differ slightly every time it is run. The random seed is set to avoid
    # this randomness. However this is not necessary for your own applications.
    seed = 42
    np.random.seed = seed

    def conv_block(x, size, dropout):
        # Convolutional layer.
        conv = layers.Conv3D(size, (3, 3, 3), kernel_initializer='he_uniform', padding="same")(x)
        conv = layers.Activation("relu")(conv)
        conv = layers.Conv3D(size, (3, 3, 3), kernel_initializer='he_uniform', padding="same")(conv)
        conv = layers.Activation("relu")(conv)
        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)
        return conv

    def gating_signal(input, out_size):
        # resize the down layer feature map into the same dimension as the up layer feature map
        # using 1x1 conv
        # :return: the gating feature map with the same dimension of the up layer feature map
        x = layers.Conv3D(out_size, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(input)
        x = layers.Activation('relu')(x)
        return x

    def attention_block(x, gating, inter_shape):
        shape_x = K.int_shape(x)  # (None, 8, 8, 8, 128)
        shape_g = K.int_shape(gating)  # (None, 4, 4, 4, 128)
        # Getting the x signal to the same shape as the gating signal
        theta_x = layers.Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), kernel_initializer='he_uniform', padding='same')(
            x)  # 16
        shape_theta_x = K.int_shape(theta_x)
        # Getting the gating signal to the same number of filters as the inter_shape
        phi_g = layers.Conv3D(inter_shape, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(gating)
        upsample_g = layers.Conv3DTranspose(inter_shape, (3, 3, 3),
                                            strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2],
                                                     shape_theta_x[3] // shape_g[3]),
                                            kernel_initializer='he_uniform', padding='same')(phi_g)  # 16
        concat_xg = layers.add([upsample_g, theta_x])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv3D(1, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = layers.UpSampling3D(
            size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(
            sigmoid_xg)  # 32
        # upsample_psi = repeat_elem(upsample_psi, shape_x[4])
        upsample_psi = layers.Concatenate(axis=-1)([upsample_psi] * shape_x[4])
        y = layers.multiply([upsample_psi, x])
        result = layers.Conv3D(shape_x[4], (1, 1, 1), kernel_initializer='he_uniform', padding='same')(y)
        return result

    def Attention_UNet_3D_Model(input_shape):
        # network structure
        filter_numb = 16  # number of filters for the first layer
        inputs = layers.Input(input_shape, dtype=tf.float32)

        # Downsampling layers
        # DownRes 1, convolution + pooling
        conv_64 = conv_block(inputs, filter_numb, dropout=0.10)
        pool_32 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_64)
        # DownRes 2
        conv_32 = conv_block(pool_32, 2 * filter_numb, dropout=0.15)
        pool_16 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_32)
        # DownRes 3
        conv_16 = conv_block(pool_16, 4 * filter_numb, dropout=0.20)
        pool_8 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_16)
        # DownRes 4
        conv_8 = conv_block(pool_8, 8 * filter_numb, dropout=0.25)
        pool_4 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_8)
        # DownRes 5, convolution only

        conv_4 = conv_block(pool_4, 16 * filter_numb, dropout=0.30)

        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_8 = gating_signal(conv_4, 8 * filter_numb)
        att_8 = attention_block(conv_8, gating_8, 8 * filter_numb)
        up_8 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(conv_4)
        up_8 = layers.concatenate([up_8, att_8])
        up_conv_8 = conv_block(up_8, 8 * filter_numb, dropout=0.25)
        # UpRes 7
        gating_16 = gating_signal(up_conv_8, 4 * filter_numb)
        att_16 = attention_block(conv_16, gating_16, 4 * filter_numb)
        up_16 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_8)
        up_16 = layers.concatenate([up_16, att_16])
        up_conv_16 = conv_block(up_16, 4 * filter_numb, dropout=0.20)
        # UpRes 8
        gating_32 = gating_signal(up_conv_16, 2 * filter_numb)
        att_32 = attention_block(conv_32, gating_32, 2 * filter_numb)
        up_32 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32])
        up_conv_32 = conv_block(up_32, 2 * filter_numb, dropout=0.15)
        # UpRes 9
        gating_64 = gating_signal(up_conv_32, filter_numb)
        att_64 = attention_block(conv_64, gating_64, filter_numb)
        up_64 = layers.UpSampling3D(size=(2, 2, 2), data_format="channels_last")(up_conv_32)
        up_64 = layers.concatenate([up_64, att_64])
        up_conv_64 = conv_block(up_64, filter_numb, dropout=0.10)

        # final convolutional layer
        conv_final = layers.Conv3D(1, (1, 1, 1))(up_conv_64)
        conv_final = layers.Activation('linear')(conv_final)

        model = models.Model(inputs=[inputs], outputs=[conv_final], name="Attention_UNet_3D_Model")
        model.summary()
        return model

    class CustomModel(tf.keras.Model):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.unet = Attention_UNet_3D_Model(input_shape)  # 你需要在这里指定输入形状

        def call(self, inputs):
            return self.unet(inputs)

        def custom_loss(self, y_true, y_pred, input):
            # Define pooling parameters
            voxel_size = 4
            pool_size = [1, voxel_size, voxel_size, voxel_size, 1]
            strides = [1, voxel_size, voxel_size, voxel_size, 1]
            # Calculate the average values of every 8*8*8 grid for the first channel
            y_true_avg = tf.nn.avg_pool3d(y_true, ksize=pool_size, strides=strides,
                                          padding='VALID')
            y_pred_avg = tf.nn.avg_pool3d(y_pred, ksize=pool_size, strides=strides,
                                          padding='VALID')

            resized_size = voxel_size
            y_true_avg_resized = tf.repeat(y_true_avg, repeats=resized_size, axis=1)
            y_true_avg_resized = tf.repeat(y_true_avg_resized, repeats=resized_size, axis=2)
            y_true_avg_resized = tf.repeat(y_true_avg_resized, repeats=resized_size, axis=3)

            y_pred_avg_resized = tf.repeat(y_pred_avg, repeats=resized_size, axis=1)
            y_pred_avg_resized = tf.repeat(y_pred_avg_resized, repeats=resized_size, axis=2)
            y_pred_avg_resized = tf.repeat(y_pred_avg_resized, repeats=resized_size, axis=3)

            body_mask = input[:, :, :, :, 2:3] == 1
            weights = tf.ones_like(y_true_avg_resized)
            # for dose_i in np.arange(0, 1, 0.1):
            #     lower_bound = dose_i
            #     upper_bound = dose_i + 0.1
            #     # Create masks for the current interval
            #     lower_mask = tf.math.greater_equal(y_true_avg_resized, lower_bound)
            #     upper_mask = tf.math.less(y_true_avg_resized, upper_bound)
            #     interval_mask = tf.math.logical_and(lower_mask, upper_mask)
            #     # Combine with body_mask
            #     mask = tf.math.logical_and(interval_mask, body_mask)
            #     # Calculate new weights for this interval and apply them
            #     interval_weights = tf.constant(100 * (1 - dose_i), dtype=tf.float32)
            #     weights = tf.where(mask, interval_weights, weights)

            # Calculate the Huber loss of the average values
            log_cosh = tf.keras.losses.LogCosh()
            log_cosh_loss_mean = weights * log_cosh(y_true_avg_resized, y_pred_avg_resized)

            log_cosh_loss_body = weights * log_cosh(y_true[input[:, :, :, :, 2:3] == 1],
                                                    y_pred[input[:, :, :, :, 2:3] == 1])
            log_cosh_loss_skin = log_cosh(y_true[input[:, :, :, :, 3:4] == 1], y_pred[input[:, :, :, :, 3:4] == 1])

            # 总损失
            log_cosh_total_loss = 8 * log_cosh_loss_mean + 1 * log_cosh_loss_body + 1 * log_cosh_loss_skin

            return log_cosh_total_loss

        def train_step(self, data):
            # Unpack the data
            x, y = data

            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = self(x, training=True)  # Compute predictions

                # Compute our own loss
                loss = tf.reduce_mean(self.custom_loss(y, y_pred, x))  # 你的自定义损失函数可以在这里使用额外的输入

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`.
            self.compiled_metrics.update_state(y, y_pred)

            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            # return {m.name: m.result() for m in self.metrics}
            # 在你的 train_step 函数中
            return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}

        def test_step(self, data):
            # Unpack the data
            x, y = data

            # Compute predictions
            y_pred = self(x, training=False)

            # Compute our own loss
            loss = tf.reduce_mean(self.custom_loss(y, y_pred, x))  # Use tf.reduce_mean to ensure a scalar loss value

            # Update the metrics.
            self.compiled_metrics.update_state(y, y_pred)

            # Return a dict mapping metric names to current value.
            return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}

    # endregion
    # endregion
    ###############################################################################
    # 3. TRAIN AND VALIDATE THE CNN MODEL #########################################
    ###############################################################################
    # region
    if mode == 'train':
        # region
        def data_generator(dose_type, start, end, batch_size):
            while True:
                for i in range(start, end, batch_size):
                    X, Y = data_load(dose_type, i, i+batch_size)
                    yield X, Y

        # 创建训练数据生成器
        train_gen = data_generator(dose_type, 1, 91, batch_size)
        # 包装为 tf.data.Dataset 对象
        train_data_gen = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 128, 128, 128, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 128, 128, 128, 1), dtype=tf.float32)
            )
        )
        # 创建验证数据生成器
        val_gen = data_generator(dose_type, 91, 100, batch_size)
        # 包装为 tf.data.Dataset 对象
        val_data_gen = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 128, 128, 128, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 128, 128, 128, 1), dtype=tf.float32)
            )
        )

        metrics = ['accuracy', 'mae']
        loss = 'mean_squared_error'
        LR = 0.001
        optimizer = tf.keras.optimizers.Adam(LR)

        # model = UNet_3D_Model(input_shape=input_shape)
        model = CustomModel()

        model.compile(optimizer=Adam(learning_rate=0.001), loss=model.custom_loss, metrics=['accuracy', 'mae'])
        # model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['accuracy', 'mae'])

        ## TO PREVENT OVERFITTING: Use early stopping method to solve model over-fitting problem
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', verbose=1)
        # The patience parameter is the amount of epochs to check for improvement

        # Checkpoint: ModelCheckpoint callback saves a model at some interval.
        checkpoint_filepath = path + '/UNet_best_model.epoch{epoch:02d}-loss{val_loss:.5f}.hdf5'
        # checkpoint_filepath = 'saved_model/Att_UNet_best_model.epoch{epoch:02d}-loss{val_loss:.5f}.hdf5'

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode='min',  # #Use Mode = max for accuracy and min for loss.
                                                        )

        # # Decaying learning rate
        # reduce_lr = tf.keras.callbacks.callback_reduce_lr_on_plateau(
        #     monitor = "val_loss",
        #     factor = 0.1,
        #     patience = 10,
        #     verbose = 0,
        #     mode = c("auto", "min", "max"),
        #     min_delta = 1e-04,
        #     cooldown = 0,
        #     min_lr = 0)

        ## CSVLogger logs epoch, acc, loss, val_acc, val_loss
        log_csv = tf.keras.callbacks.CSVLogger(path + '_log.csv', separator=',', append=False)

        # Train the model
        import time
        start = time.time()
        # start1 = datetime.now()
        # 然后在model.fit中使用这些生成器
        history = model.fit(train_data_gen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[early_stopping, checkpoint, log_csv],
                            validation_data=val_data_gen,
                            validation_steps=val_steps_per_epoch,
                            )


        finish = time.time()
        # stop = datetime.now()
        # Execution time of the model
        print('total execution time in seconds is: ', finish - start)
        # print(history.history.keys())
        print('Training has been finished successfully')

        ## Plot training history
        ## LEARNING CURVE: plots the graph of the training loss vs.validation
        # loss over the number of epochs.
        def plot_history(history):
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('average training loss and validation loss')
            plt.ylabel('mean-squared error')
            plt.xlabel('epoch')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(['training loss', 'validation loss'], loc='upper right')
            # plt.ylim([0, 50])  # 设置y轴的范围
            plt.show()
        # plot_history(history)

        # Save Model
        model.save_weights(path + '/Unet3D_model_120epochs.h5')

        # # Evaluating the model
        # train_loss, train_acc, train_acc1 = model.evaluate(X_train, Y_train, batch_size = 8)
        # val_loss, test_acc, train_acc2 = model.evaluate(X_val, Y_val, batch_size = 8)
        # print('Train: %.3f, Test: %.3f' % (train_loss, val_loss))
        # endregion
    # endregion
    ###############################################################################
    # 4. MAKE PREDICTIONS ON TEST DATASET #########################################
    ###############################################################################
    # region
    if mode == 'predict':
        # region
        from scipy.ndimage import zoom
        dose_merge_ratio = 4
        pre_data_range = [157, 171]
        X_test, Y_test = data_load(dose_type, pre_data_range[0], pre_data_range[1])
        new_model = CustomModel()
        # 使用虚拟输入调用模型
        dummy_input = np.zeros((1, *input_shape))
        _ = new_model(dummy_input)
        new_model.load_weights(path + '/' + model_name)
        # new_model = load_model('saved_model/Unet3D_model_120epochs.h5')
        # Check its architecture
        new_model.summary()
        #  Predict on the test set: Let us see how the model generalize by using the test set.
        predict_test = new_model.predict(X_test, verbose=1, batch_size=2)
        # Processing the negative values
        for i in range(X_test.shape[0]):
            nor_factor = 0.005531000000000002
            print('predict_test.shape:', predict_test.shape)
            predict = predict_test[i, :, :, :, 0]

            predict[predict < 0] = 0
            Y = np.load(f'../../PHITS_data_processed_for_3DUnet_z_up/1_ratio/GLI_{str(i + pre_data_range[0]).zfill(3)}_GBM_dose_data/data/nitrogendetail_doses.npy')

            predict = predict * nor_factor
            Y = Y * nor_factor
            predict = np.transpose(predict[:, :, :], (2, 0, 1))
            patient_number = str(i + pre_data_range[0]).zfill(3)
            rbe_path = os.path.join('../../PHITS_data', f'GLI_{patient_number}_GBM_dose_data', '1_RBE_and_ROI_mask')
            rbe = np.loadtxt(rbe_path + '/rbe_n_128.txt').reshape((128, 128, 128))
            ppm_path = os.path.join('../../PHITS_data', f'GLI_{patient_number}_GBM_dose_data', '2_PPM_PER')
            ppm = np.loadtxt(ppm_path + '/n_per_128.txt').reshape((128, 128, 128))

            predict = predict * rbe * ppm*40
            Y = Y * rbe * ppm*40
            predict = predict .astype(np.float32)
            Y = Y.astype(np.float32)
            new_dir = '../../PHITS_data_processed_for_3DUnet_z_up/'+str(dose_merge_ratio)+'_ratio'
            save_path = os.path.join(new_dir, f'GLI_{patient_number}_GBM_dose_data', 'data')
            print(save_path)
            np.save(save_path + "/coupling_" + dose_type + "_.npy", predict)
            print(predict.shape)
            np.save(save_path + "/monte_" + dose_type + "_.npy", Y)
            print(Y.shape)
        # endregion
    # endregion

