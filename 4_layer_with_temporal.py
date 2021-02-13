def grad_cam(model, image, category_index, layer_name):
    '''model = Sequential()
    model.add(input_model)'''

    nb_classes = 487
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    #model.add(Lambda(target_layer,output_shape = target_category_loss_output_shape))

    n_model = tf.keras.Model(inputs= model.input, outputs= model.layers[11].output)
    

    with tf.GradientTape() as tape:
      output, conv_output = m4(image)
      class_output=tf.reduce_sum(target_category_loss(output,category_index,nb_classes))
      #conv_output = n_model(image)
      #loss = K.sum([model.layers[-1].output])
      #conv_output =  model.layers[-9].output
    variables = model.trainable_variables
    grads = (tape.gradient(class_output, conv_output))
    weights = grads[0]
    # modify here for more weights
    w0 = normalize(weights[0])
    w1 = normalize(weights[1])
    w2 = normalize(weights[2])
    w3 = normalize(weights[3])

    print(np.shape(grads))
    
    
    output = conv_output[0]
    print(np.shape(output))
    
    #weights = np.mean(weights, axis = (1, 2))
    w0 = np.mean(w0, axis = (0, 1))
    w1 = np.mean(w1, axis = (0, 1))
    w2 = np.mean(w2, axis = (0, 1))
    w3 = np.mean(w3, axis = (0, 1))
    print(np.shape(weights))
    cam1 = np.ones(output.shape[1 : 3], dtype = np.float32)
    cam2 = np.ones(output.shape[1 : 3], dtype = np.float32)
    cam3 = np.ones(output.shape[1 : 3], dtype = np.float32)
    cam4 = np.ones(output.shape[1 : 3], dtype = np.float32)
  
    for i in range(512):
        cam1 += w0[i] * output[0,:, :, i]
        cam2 += w1[i] * output[1,:, :, i]
        cam3 += w2[i] * output[2,:, :, i]
        cam4 += w3[i] * output[3,:, :, i]


    cam1 = cv2.resize(np.float32(cam1), (112, 112), cv2.INTER_AREA)
    cam1 = np.maximum(cam1, 0)
    heatmap1 = cam1 / np.max(cam1)

    cam2 = cv2.resize(np.float32(cam2), (112, 112), cv2.INTER_AREA)
    cam2 = np.maximum(cam2, 0)
    heatmap2 = cam2 / np.max(cam2)

    cam3 = cv2.resize(np.float32(cam3), (112, 112), cv2.INTER_AREA)
    cam3 = np.maximum(cam3, 0)
    heatmap3 = cam3 / np.max(cam3)

    cam4 = cv2.resize(np.float32(cam4), (112, 112), cv2.INTER_AREA)
    cam4 = np.maximum(cam4, 0)
    heatmap4 = cam4 / np.max(cam4)

    print(np.shape(image))
    #Return to BGR [0..255] from the preprocessed image
    image = image[0]
    img1, img2 = image[1], image[4]
    img1 -= np.min(img1)
    img1 = np.minimum(img1, 255)
    img2 -= np.min(img2)
    img2 = np.minimum(img2, 255)
    img3, img4 = image[8], image[11]
    img3 -= np.min(img3)
    img3 = np.minimum(img3, 255)
    img4 -= np.min(img4)
    img4 = np.minimum(img4, 255)

    cam1 = cv2.applyColorMap(np.uint8(255*heatmap1), cv2.COLORMAP_JET)
    cam1 = np.float32(cam1) + np.float32(img1)
    cam1 = 255 * cam1 / np.max(cam1)

    cam2 = cv2.applyColorMap(np.uint8(255*heatmap2), cv2.COLORMAP_JET)
    cam2 = np.float32(cam2) + np.float32(img2)
    cam2 = 255 * cam2 / np.max(cam2)

    cam3 = cv2.applyColorMap(np.uint8(255*heatmap3), cv2.COLORMAP_JET)
    cam3 = np.float32(cam3) + np.float32(img3)
    cam3 = 255 * cam3 / np.max(cam3)

    cam4 = cv2.applyColorMap(np.uint8(255*heatmap4), cv2.COLORMAP_JET)
    cam4 = np.float32(cam4) + np.float32(img4)
    cam4 = 255 * cam4 / np.max(cam4)
    cam1, cam2, cam3, cam4 = cv2.cvtColor(cam1, cv2.COLOR_BGR2RGB), cv2.cvtColor(cam2, cv2.COLOR_BGR2RGB), cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB), cv2.cvtColor(cam4, cv2.COLOR_BGR2RGB)
    return np.uint8(cam1), heatmap1, np.uint8(cam2), heatmap2, np.uint8(cam3), heatmap3, np.uint8(cam4), heatmap4
#predictions = c3d_model.predict(preprocessed_input)
#top_1 = decode_predictions(predictions)[0][0]
#print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
predictions = m2.predict(inp1)
predicted_class = np.argmax(predictions)
cam1, heatmap1, cam2, heatmap2,  cam3, heatmap3, cam4, heatmap4 = grad_cam(m1, inp1, predicted_class, "conv5b")
fol = 'gradcam_4_' + str(folder_name)
import matplotlib.pyplot as plt
import skimage
import skimage.io as imshow
plt.figure(figsize = (8,8))
plt.subplot(1,4,1)
plt.imshow(cam1)
plt.subplot(1,4,2)
plt.imshow(cam2)
plt.subplot(1,4,3)
plt.imshow(cam3)
plt.subplot(1,4,4)
plt.imshow(cam4)
f_name = '/content/' + str(fol) + '.png'
plt.savefig(f_name) 
FILE.download(f_name)
#!rm -r {unzip_name}
