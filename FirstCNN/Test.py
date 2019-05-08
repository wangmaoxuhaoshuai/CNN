from Predict import imageprepare, predictInt

if __name__ == '__main__':
    for i in range(10):
        imvalue = imageprepare('Test_data/%d.jpg' % (int(i)))
        predint = predictInt(imvalue)
        print(predint[0])